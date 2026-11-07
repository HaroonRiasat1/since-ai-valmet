"""Data semantic layer: find, describe and query plant data. The model fills a spec, it never writes SQL."""
import difflib
import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import yaml

from backend.config import get_settings

MAX_CHARS = 6000
DEFAULT_LIMIT = 50
MAX_LIMIT = 200
SEARCH_HITS = 8
AGGREGATES = {"sum", "avg", "min", "max", "count"}
OPERATORS = {"=", "!=", ">", "<", ">=", "<=", "like", "in", "between"}
BONUS = {"entity": 2, "recipe": 2, "glossary": 1, "column": 0}


class SpecError(ValueError):
    """A query spec the layer cannot honour. The message tells the model how to fix it."""


def tokens(text):
    """Lower-case words with the plural s removed."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words if len(w) > 1}


def fit(value):
    """Every tool result stays small enough for a small model's context."""
    text = json.dumps(value, default=str)
    if len(text) <= MAX_CHARS:
        return value
    if isinstance(value, dict) and value.get("rows"):
        rows = list(value["rows"])
        while len(json.dumps({**value, "rows": rows}, default=str)) > MAX_CHARS - 100 and len(rows) > 1:
            rows = rows[:max(1, len(rows) * 3 // 4)]
        return {**value, "rows": rows, "note": f"showing {len(rows)} rows of {len(value['rows'])}"}
    return {"note": "cut to fit, ask for something narrower", "text": text[:MAX_CHARS - 100]}


class Layer:
    def __init__(self, folder: Path):
        data = yaml.safe_load((Path(folder) / "layer.yaml").read_text(encoding="utf8"))
        self.db = Path(folder).resolve() / "plant.sqlite"
        self.entities = {e["name"]: {**e, "cols": {c["name"]: c for c in e["columns"]}} for e in data["entities"]}
        self.recipes = {r["name"]: r for r in data["recipes"]}
        self.glossary = {g["term"].lower(): g for g in data["glossary"]}
        for entity in self.entities.values():
            entity["aliases"] = {alias.lower(): c["name"] for c in entity["columns"] for alias in (c["label"], *c.get("synonyms", []))}

    # ---------- finding things
    def _items(self):
        for e in self.entities.values():
            yield "entity", e["name"], [e["name"], e["label"], *e.get("synonyms", [])], e["description"], e["label"]
            for c in e["cols"].values():
                yield "column", f"{e['name']}.{c['name']}", [c["name"], c["label"], *c.get("synonyms", [])], c.get("description", ""), c["label"]
        for r in self.recipes.values():
            yield "recipe", r["name"], [r["name"], r["label"]], r["description"], r["label"]
        for g in self.glossary.values():
            yield "glossary", g["term"], [g["term"]], g["meaning"], g["term"]

    def search(self, query: str) -> list[dict]:
        """Best matches for a business phrase. A column hit also lifts its entity."""
        wanted = tokens(query)
        scored, lift = {}, {}
        for kind, name, labels, text, label in self._items():
            hit = len(wanted & tokens(" ".join(labels)))
            score = 3 * hit + len(wanted & tokens(text)) + (BONUS[kind] if hit else 0)
            if kind == "column" and hit:
                entity = name.split(".")[0]
                weight = 6 if self.entities[entity]["cols"][name.split(".")[1]]["role"] == "measure" else 1   # a measure is what the question is about
                lift[entity] = min(lift.get(entity, 0) + hit * weight, 12)
            if score:
                scored[name] = [score, {"name": name, "kind": kind, "label": label, "description": text[:120]}]
        for entity, extra in lift.items():
            e = self.entities[entity]
            scored.setdefault(entity, [1, {"name": entity, "kind": "entity", "label": e["label"], "description": e["description"][:120]}])[0] += extra
        return [hit for _, hit in sorted(scored.values(), key=lambda s: -s[0])[:SEARCH_HITS]]

    def card(self, entity: dict) -> dict:
        columns = entity["cols"].values()
        out = {"entity": entity["name"], "label": entity["label"], "grain": entity["grain"], "main_date": entity.get("main_date"),
               "measures": [f"{c['name']} ({c['label']}, {c['agg']}{', ' + c['unit'] if c.get('unit') else ''})" for c in columns if c["role"] == "measure"],
               "dimensions": [c["name"] for c in columns if c["role"] != "measure"],
               "joins": {j["name"]: j["to"] for j in entity.get("joins", [])},
               "hint": "Joined columns are '<join>.<column>'. Use 'count' as a measure to count rows."}
        for key in ("rules", "required_filters"):
            if entity.get(key):
                out[key] = entity[key]
        return out

    def describe(self, name: str) -> dict:
        if name in self.entities:
            return fit(self.card(self.entities[name]))
        if "." in name:
            entity, column = name.split(".", 1)
            found = self.entities.get(entity, {}).get("cols", {}).get(column)
            if found:
                return fit({"entity": entity, **found})
        if name in self.recipes:
            r = self.recipes[name]
            return {k: r[k] for k in ("name", "label", "description", "parameters")}
        if name.lower() in self.glossary:
            return self.glossary[name.lower()]
        return {"error": f"'{name}' not found, try search_layer"}

    # ---------- compiling a spec
    def _field(self, entity: dict, field: str, joins: dict) -> tuple[str, dict]:
        """Resolve 'column' or 'join.column' to checked SQL. Only names in the layer ever reach the query."""
        if "." not in field:
            alias, target = "t", entity
        else:
            head, field = field.split(".", 1)
            join = next((j for j in entity.get("joins", []) if head in (j["name"], j["to"])), None)
            if join is None:
                raise SpecError(f"{entity['name']} has no join '{head}'. Joins: {[j['name'] for j in entity.get('joins', [])]}")
            alias, target = join["name"], self.entities[join["to"]]
            joins[join["name"]] = join
        field = target["aliases"].get(field.lower(), field)
        column = target["cols"].get(field)
        if column is None:
            near = difflib.get_close_matches(field, target["cols"], n=3, cutoff=0.5)
            raise SpecError(f"{target['name']} has no column '{field}'. Did you mean {near or list(target['cols'])[:12]}?")
        return f'{alias}."{field}"', column

    @staticmethod
    def _value(column: dict, value):
        """Correct the case of a known value, and refuse one that does not exist."""
        allowed = column.get("values")
        if not allowed or not isinstance(value, str):
            return value
        match = next((v for v in allowed if v.lower() == value.lower()), None)
        if match is None:
            raise SpecError(f"{column['name']} has no value '{value}'. Values: {allowed}")
        return match

    def _measure(self, entity, measure, joins):
        if isinstance(measure, str):
            measure = {"column": measure}
        name, agg = measure.get("column"), measure.get("agg")
        if name == "count":
            return "COUNT(*)", "count"
        sql, column = self._field(entity, name, joins)
        name = sql.replace("t.", "", 1).replace('"', "").replace(".", "_")
        if column["role"] != "measure":
            raise SpecError(f"{name} is not a measure. Put it in dimensions.")
        agg = (agg or column["agg"]).lower()
        if agg not in AGGREGATES:
            raise SpecError(f"unknown aggregate {agg}. Use {sorted(AGGREGATES)}")
        return f"{agg.upper()}({sql})", f"{agg}_{name}"

    def _filter(self, entity, item, joins):
        field, op, value = item.get("field"), str(item.get("op", "=")).lower(), item.get("value")
        if op not in OPERATORS:
            raise SpecError(f"unknown operator {op}. Use {sorted(OPERATORS)}")
        sql, column = self._field(entity, field, joins)
        if op in ("in", "between"):
            values = [self._value(column, v) for v in (value if isinstance(value, list) else [value])]
            if op == "between" and len(values) != 2:
                raise SpecError("between needs [from, to]")
            if op == "between":
                return f"{sql} BETWEEN ? AND ?", values
            return f"{sql} IN ({', '.join('?' * len(values))})", values
        return f"{sql} {op.upper()} ?", [self._value(column, value)]

    def compile(self, spec: dict) -> tuple[str, list]:
        entity = self.entities.get(spec.get("entity"))
        if entity is None:
            raise SpecError(f"unknown entity {spec.get('entity')}. Entities: {list(self.entities)}")
        joins, select, names, group, where, params = {}, [], [], [], [], []
        for dimension in spec.get("dimensions") or []:
            sql, _ = self._field(entity, dimension, joins)
            name = dimension.replace(".", "_")
            select.append(f'{sql} AS "{name}"')
            names.append(name)
            group.append(sql)
        measures = spec.get("measures") or []
        for measure in measures:
            sql, name = self._measure(entity, measure, joins)
            select.append(f'{sql} AS "{name}"')
            names.append(name)
        if not select:
            raise SpecError("give at least one measure or dimension")
        filters = spec.get("filters") or []
        used = {entity["aliases"].get(str(f.get("field")).lower(), f.get("field")) for f in filters}
        for need in entity.get("required_filters", []):
            if not used & set(need["any_of"]):
                raise SpecError(f"{need['message']} Filter on one of {need['any_of']}.")
        for item in filters:
            sql, values = self._filter(entity, item, joins)
            where.append(sql)
            params += values
        query = f'SELECT {", ".join(select)} FROM "{entity["name"]}" t'
        for name, join in joins.items():
            left, right = join["keys"]
            query += f' LEFT JOIN "{join["to"]}" {name} ON t."{left}" = {name}."{right}"'
        if where:
            query += " WHERE " + " AND ".join(where)
        if measures and group:
            query += " GROUP BY " + ", ".join(group)
        order = []
        for item in spec.get("order_by") or []:
            field = str(item.get("field", "")).replace(".", "_")
            match = next((n for n in names if n == field or n.endswith(f"_{field}")), None)
            if match is None:
                raise SpecError(f"order_by must use a selected field: {names}")
            order.append(f'"{match}" {"DESC" if str(item.get("dir", "asc")).lower() == "desc" else "ASC"}')
        if order:
            query += " ORDER BY " + ", ".join(order)
        return f"{query} LIMIT {min(int(spec.get('limit') or DEFAULT_LIMIT), MAX_LIMIT)}", params

    # ---------- running
    def run(self, spec: dict) -> dict:
        sql, params = self.compile(spec)
        with sqlite3.connect(f"{self.db.as_uri()}?mode=ro", uri=True) as db:
            cursor = db.execute(sql, params)
            columns = [c[0] for c in cursor.description]
            rows = [[round(v, 2) if isinstance(v, float) else v for v in row] for row in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "row_count": len(rows), "sql": sql, "params": params}

    def term(self, message: str) -> dict | None:
        """A glossary entry if the message asks what a term means."""
        asked = re.search(r"\b(?:what (?:does|is|are)|meaning of|define)\s+(?:an? |the )?([\w -]+?)(?:\s+mean)?\??$", message.strip(), re.I)
        return self.glossary.get(asked.group(1).lower()) if asked else None

    def report(self, name: str, params: dict | None = None) -> dict:
        recipe = self.recipes.get(name)
        if recipe is None:
            raise SpecError(f"unknown report {name}. Reports: {list(self.recipes)}")
        params = params or {}
        missing = [p["name"] for p in recipe["parameters"] if p.get("required") and p["name"] not in params]
        if missing:
            raise SpecError(f"report {name} needs {missing}")
        text = json.dumps(recipe["spec"])
        for key, value in params.items():
            text = text.replace("{" + key + "}", json.dumps(str(value))[1:-1])
        return self.run(json.loads(text))


@lru_cache
def get_layer() -> Layer:
    return Layer(get_settings().semantic_dir)

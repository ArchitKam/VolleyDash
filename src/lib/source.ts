/**
 * A workspace's data source. Two grains exist:
 *  - "measure": one row is a player's whole-match totals (recruiting CSV exports)
 *  - "event":   one row is a single action (DataVolley .dvw play-by-play)
 * Every schema vocabulary here is DERIVED from the loaded files, never hardcoded
 * from a format specification — a validator rejects a value because it wasn't
 * seen in these files, and can say which values were.
 */

export type Grain = "measure" | "event";

export type FieldRole = "identity" | "dimension" | "measure";

export type FieldSpec = {
  name: string;
  role: FieldRole;
  description: string;
  /** null = open domain (not enumerable, so not offered as a filter value list). */
  values?: Set<string> | null;
};

export type SourceSchema = {
  fields: Record<string, FieldSpec>;
  /**
   * Legal dependent-field values, keyed `${governingField}|${dependentField}`
   * then by the governing value. An evaluation code means something different
   * per skill and isn't legal for every skill, so codes are never a flat list.
   */
  dependentValues: Map<string, Map<string, Set<string>>>;
};

export type FactRow = Record<string, string | number | null>;

export type MeasureRow = {
  Player: string;
  Game: string;
  Set?: string | undefined;
  measure: string;
  value: number;
};

export interface Source {
  grain: Grain;
  schema: SourceSchema;
  /** The rows themselves, at this source's grain. */
  facts(): FactRow[];
  /** Axis name -> this source's own column name. */
  identityFields(): Record<string, string>;
  /** Identity axes plus "Metric". */
  axes(): string[];
  /** Per-identity quantities not derivable by counting facts rows. */
  measures(): MeasureRow[];
  measureAggregation(name: string): "sum" | "mean";
  gameLabels(): string[];
  setLabels(): string[];
  playerLabels(): string[];
  warnings(): string[];
}

export function dependentKey(governing: string, dependent: string): string {
  return `${governing}|${dependent}`;
}

export function dependentAllowed(
  schema: SourceSchema,
  governing: string,
  governingValue: string,
  dependent: string,
): Set<string> | null {
  return schema.dependentValues.get(dependentKey(governing, dependent))?.get(governingValue) ?? null;
}

export function supportsSets(source: Source): boolean {
  return Object.keys(source.identityFields()).includes("Set");
}

/** Records an observed (governing value -> dependent value) pair while deriving a schema. */
export function noteDependent(
  schema: SourceSchema,
  governing: string,
  governingValue: string,
  dependent: string,
  dependentValue: string,
): void {
  const key = dependentKey(governing, dependent);
  const byGoverning = schema.dependentValues.get(key) ?? new Map<string, Set<string>>();
  const set = byGoverning.get(governingValue) ?? new Set<string>();
  set.add(dependentValue);
  byGoverning.set(governingValue, set);
  schema.dependentValues.set(key, byGoverning);
}

export function emptySchema(): SourceSchema {
  return { fields: {}, dependentValues: new Map() };
}

export function fieldValueList(schema: SourceSchema, field: string): string[] {
  const values = schema.fields[field]?.values;
  return values ? [...values].sort((a, b) => a.localeCompare(b)) : [];
}

export function filterableFields(schema: SourceSchema): FieldSpec[] {
  return Object.values(schema.fields).filter((f) => f.role === "dimension");
}

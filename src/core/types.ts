export const enum CellType {
  Empty,
  Food,
  Wall,
  Mouth,
  Producer,
  Mover,
  Killer,
  Armor,
}

export interface LocalCell {
  type: CellType;
  x: number;
  y: number;
}

export interface Metrics {
  organisms: number;
  record: number;
  generation: number;
  ticks: number;
  largest: number;
  averageMutation: number;
}

export const DIRECTIONS = [[0, -1], [0, 1], [-1, 0], [1, 0]] as const;

export const CELL_COLORS: Record<CellType, string> = {
  [CellType.Empty]: "#090506",
  [CellType.Food]: "#fda4af",
  [CellType.Wall]: "#3f1119",
  [CellType.Mouth]: "#fb7185",
  [CellType.Producer]: "#fecdd3",
  [CellType.Mover]: "#ef4444",
  [CellType.Killer]: "#7f1d1d",
  [CellType.Armor]: "#be123c",
};

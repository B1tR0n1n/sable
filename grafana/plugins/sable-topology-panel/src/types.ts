export interface TopologyOptions {
  showLabels: boolean;
  showConfidence: boolean;
  animatePulse: boolean;
  layout: 'tiered' | 'grid';
}

export interface NodeData {
  index: number;
  id: string;
  label: string;
  type: string;
  tier: string;
  state: string;
  stateIdx: number;
  confidence: number;
}

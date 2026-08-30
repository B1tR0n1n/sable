import React, { useMemo } from 'react';
import { PanelProps, DataFrame } from '@grafana/data';
import { TopologyOptions, NodeData } from './types';

// b1tr0n1n color palette
const COLORS = {
  bg: '#0a0908',
  panel: '#0f0e0b',
  border: '#2a2520',
  text: '#c8bda0',
  dim: '#7a7060',
  bright: '#ede5d0',
  gold: '#c9a227',
  dimGold: '#8b7320',
  success: '#4a7a45',
  danger: '#a63d2f',
};

const STATE_COLORS: Record<string, { bg: string; text: string; border: string; glow?: string }> = {
  healthy: {
    bg: 'rgba(74, 122, 69, 0.25)',
    text: COLORS.success,
    border: COLORS.border,
  },
  degraded: {
    bg: 'rgba(201, 162, 39, 0.25)',
    text: COLORS.gold,
    border: COLORS.border,
  },
  failed: {
    bg: 'rgba(166, 61, 47, 0.40)',
    text: COLORS.danger,
    border: COLORS.danger,
    glow: 'rgba(166, 61, 47, 0.5)',
  },
  unreachable: {
    bg: COLORS.bg,
    text: COLORS.dim,
    border: COLORS.border,
  },
  unknown: {
    bg: COLORS.panel,
    text: COLORS.dim,
    border: COLORS.border,
  },
};

// CSS keyframes injected once
const PULSE_CSS = `
@keyframes sable-pulse-danger {
  0%, 100% { box-shadow: 0 0 8px rgba(166,61,47,0.4); }
  50% { box-shadow: 0 0 20px rgba(166,61,47,0.7), inset 0 0 12px rgba(166,61,47,0.3); }
}
`;

function parseFrameToNodes(data: DataFrame[]): NodeData[] {
  if (!data || data.length === 0) return [];
  const frame = data[0];
  const nodes: NodeData[] = [];

  const len = frame.fields[0]?.values?.length ?? 0;
  const getField = (name: string) => frame.fields.find((f) => f.name === name);

  for (let i = 0; i < len; i++) {
    nodes.push({
      index: getField('Index')?.values[i] ?? i,
      id: getField('ID')?.values[i] ?? `node-${i}`,
      label: getField('Label')?.values[i] ?? `Node ${i}`,
      type: getField('Type')?.values[i] ?? 'UNKNOWN',
      tier: getField('Tier')?.values[i] ?? '',
      state: (getField('State')?.values[i] ?? 'unknown').toLowerCase(),
      stateIdx: getField('StateIdx')?.values[i] ?? -1,
      confidence: getField('Confidence')?.values[i] ?? 0,
    });
  }

  return nodes;
}

interface Props extends PanelProps<TopologyOptions> {}

export const TopologyPanel: React.FC<Props> = ({ data, options, width, height }) => {
  const nodes = useMemo(() => parseFrameToNodes(data.series), [data.series]);

  // Group nodes by tier for tiered layout
  const tiers = useMemo(() => {
    if (options.layout !== 'tiered') return null;
    const tierMap: Record<string, NodeData[]> = {};
    for (const node of nodes) {
      const tier = node.tier || 'Other';
      if (!tierMap[tier]) tierMap[tier] = [];
      tierMap[tier].push(node);
    }
    // Sort tiers: network > compute > storage > application > other
    const tierOrder = ['network', 'compute', 'storage', 'application', 'database', 'other'];
    return Object.entries(tierMap).sort(([a], [b]) => {
      const ai = tierOrder.indexOf(a.toLowerCase());
      const bi = tierOrder.indexOf(b.toLowerCase());
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
    });
  }, [nodes, options.layout]);

  // Calculate cell size based on available space
  const maxPerRow = Math.max(4, Math.ceil(Math.sqrt(nodes.length)));
  const cellSize = Math.min(
    Math.floor((width - 40) / maxPerRow) - 6,
    Math.floor((height - 80) / (tiers ? tiers.length * 1.5 : Math.ceil(nodes.length / maxPerRow))) - 6,
    80
  );
  const clampedSize = Math.max(36, Math.min(cellSize, 80));

  const renderNode = (node: NodeData) => {
    const sc = STATE_COLORS[node.state] || STATE_COLORS.unknown;
    const isFailed = node.state === 'failed' && options.animatePulse;

    return (
      <div
        key={node.index}
        title={`${node.label} (${node.type})\nState: ${node.state}\nConfidence: ${(node.confidence * 100).toFixed(1)}%`}
        style={{
          width: clampedSize,
          height: clampedSize,
          background: sc.bg,
          border: `1px solid ${sc.border}`,
          borderRadius: 3,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          transition: 'border-color 0.2s, background 0.3s',
          animation: isFailed ? 'sable-pulse-danger 1.5s ease-in-out infinite' : undefined,
          borderStyle: node.state === 'unreachable' ? 'dashed' : 'solid',
          position: 'relative',
        }}
      >
        {options.showLabels && (
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: clampedSize > 50 ? 9 : 7,
              color: COLORS.dim,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              maxWidth: clampedSize - 8,
              textAlign: 'center',
            }}
          >
            {node.label}
          </span>
        )}
        <span
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: clampedSize > 50 ? 7 : 6,
            color: sc.text,
            fontWeight: node.state === 'failed' ? 600 : 400,
            marginTop: 2,
            textTransform: 'uppercase',
          }}
        >
          {node.state}
        </span>
        {options.showConfidence && clampedSize > 50 && (
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 7,
              color: COLORS.dim,
              marginTop: 1,
            }}
          >
            {(node.confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
    );
  };

  return (
    <div
      style={{
        width,
        height,
        background: COLORS.panel,
        overflow: 'auto',
        padding: 12,
        fontFamily: "'JetBrains Mono', monospace",
      }}
    >
      <style>{PULSE_CSS}</style>

      {tiers ? (
        // Tiered layout
        tiers.map(([tierName, tierNodes]) => (
          <div key={tierName} style={{ marginBottom: 8 }}>
            <div
              style={{
                fontSize: 7,
                letterSpacing: 3,
                color: COLORS.dim,
                textTransform: 'uppercase',
                textAlign: 'center',
                marginBottom: 4,
                opacity: 0.6,
              }}
            >
              {tierName}
            </div>
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: 4,
                justifyContent: 'center',
              }}
            >
              {tierNodes.map(renderNode)}
            </div>
            <div
              style={{
                borderTop: `1px solid ${COLORS.border}`,
                opacity: 0.3,
                margin: '6px 0',
              }}
            />
          </div>
        ))
      ) : (
        // Grid layout
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 4,
            justifyContent: 'center',
          }}
        >
          {nodes.map(renderNode)}
        </div>
      )}
    </div>
  );
};

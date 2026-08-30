import { PanelPlugin } from '@grafana/data';
import { TopologyPanel } from './TopologyPanel';
import { TopologyOptions } from './types';

export const plugin = new PanelPlugin<TopologyOptions>(TopologyPanel).setPanelOptions((builder) => {
  builder
    .addBooleanSwitch({
      path: 'showLabels',
      name: 'Show node labels',
      defaultValue: true,
    })
    .addBooleanSwitch({
      path: 'showConfidence',
      name: 'Show confidence values',
      defaultValue: true,
    })
    .addBooleanSwitch({
      path: 'animatePulse',
      name: 'Animate failed nodes',
      defaultValue: true,
    })
    .addSelect({
      path: 'layout',
      name: 'Layout',
      defaultValue: 'tiered',
      settings: {
        options: [
          { value: 'tiered', label: 'Tiered (by infrastructure tier)' },
          { value: 'grid', label: 'Grid' },
        ],
      },
    });
});

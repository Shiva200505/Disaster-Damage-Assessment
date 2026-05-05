import { DamageStats as DamageStatsType } from '../types';
import { motion } from 'framer-motion';

export default function DamageStats({ stats }: { stats: DamageStatsType }) {
  const categories = [
    { label: 'Destroyed', value: stats.destroyed, color: 'bg-red-500', text: 'text-red-700', bg: 'bg-red-50' },
    { label: 'Major Damage', value: stats.major_damage, color: 'bg-amber-500', text: 'text-amber-700', bg: 'bg-amber-50' },
    { label: 'Minor Damage', value: stats.minor_damage, color: 'bg-yellow-400', text: 'text-yellow-700', bg: 'bg-yellow-50' },
    { label: 'No Damage', value: stats.no_damage, color: 'bg-emerald-500', text: 'text-emerald-700', bg: 'bg-emerald-50' },
  ];

  return (
    <div className="space-y-4 w-full">
      <div className="flex justify-between items-end mb-4">
        <h3 className="text-lg font-bold text-slate-800">Damage Distribution</h3>
        <span className="text-sm text-slate-500 font-medium">Total: {stats.total_buildings}</span>
      </div>
      
      {categories.map((cat, i) => {
        const pct = stats.total_buildings > 0 ? (cat.value / stats.total_buildings) * 100 : 0;
        return (
          <div key={cat.label} className="space-y-1.5">
            <div className="flex justify-between text-sm">
              <span className={`font-semibold ${cat.text}`}>{cat.label}</span>
              <span className="text-slate-600 font-mono text-xs mt-0.5">{cat.value} ({pct.toFixed(1)}%)</span>
            </div>
            <div className={`w-full h-3 rounded-full overflow-hidden ${cat.bg}`}>
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 1, delay: i * 0.1, ease: 'easeOut' }}
                className={`h-full rounded-full ${cat.color}`}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

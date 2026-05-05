import { Check, Loader2 } from 'lucide-react';
import React from 'react';

const STEPS = [
  { id: 'data_loading', label: 'Loading images' },
  { id: 'preprocessing', label: 'Preprocessing' },
  { id: 'change_detection', label: 'Change detection' },
  { id: 'polygon_extraction', label: 'Finding buildings' },
  { id: 'damage_classification', label: 'Classifying damage' },
  { id: 'generating_outputs', label: 'Building map' },
];

export default function ProgressStepper({ activeStep }: { activeStep?: string }) {
  const activeIndex = activeStep ? Math.max(0, STEPS.findIndex(s => s.id === activeStep)) : 0;

  return (
    <div className="w-full py-6">
      <div className="flex flex-col gap-6">
        {STEPS.map((step, index) => {
          const isCompleted = index < activeIndex;
          const isActive = index === activeIndex;

          return (
            <div key={step.id} className="flex items-center gap-4 relative">
              {index !== STEPS.length - 1 && (
                <div className={`absolute top-10 left-5 w-0.5 h-10 -ml-px ${isCompleted ? 'bg-sky-500' : 'bg-slate-200'}`} />
              )}
              
              <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 border-2 transition-colors ${
                isCompleted ? 'bg-sky-500 border-sky-500 text-white' : 
                isActive ? 'border-sky-500 text-sky-600 bg-sky-50' : 
                'border-slate-200 text-slate-400 bg-white'
              }`}>
                {isCompleted ? <Check size={20} /> : isActive ? <Loader2 size={20} className="animate-spin" /> : <span>{index + 1}</span>}
              </div>
              
              <div className="flex flex-col">
                <span className={`font-medium ${isActive ? 'text-slate-900' : isCompleted ? 'text-slate-700' : 'text-slate-400'}`}>
                  {step.label}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

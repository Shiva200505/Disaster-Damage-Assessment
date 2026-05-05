import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { getResult } from '../api/client';
import DamageStats from '../components/DamageStats';
import MapViewer from '../components/MapViewer';
import { Download, Share2, Plus, AlertTriangle } from 'lucide-react';

export default function ResultPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();

  const { data, isLoading, isError } = useQuery({
    queryKey: ['result', taskId],
    queryFn: () => getResult(taskId!).then(res => res.data),
    retry: false
  });

  if (isLoading) {
    return <div className="min-h-screen flex items-center justify-center font-medium text-slate-500">Loading results...</div>;
  }

  if (isError || !data || data.status !== 'SUCCESS') {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-4">
          <AlertTriangle size={48} className="mx-auto text-amber-500" />
          <h2 className="text-xl font-bold">Result Not Found</h2>
          <button onClick={() => navigate('/')} className="text-sky-600 font-medium hover:underline">Start New Analysis</button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      {/* Sidebar */}
      <div className="w-[340px] shrink-0 bg-white border-r border-slate-200 flex flex-col h-full z-10 shadow-xl">
        <div className="p-5 border-b border-slate-100 flex items-center justify-between bg-white text-slate-900 sticky top-0">
          <h1 className="font-bold text-lg tracking-tight">Analysis Result</h1>
          <button 
                onClick={() => navigate('/')}
                className="p-2 hover:bg-slate-100 rounded-lg text-slate-500 transition-colors"
                title="New Analysis"
          >
            <Plus size={20} />
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-6 space-y-8">
            <DamageStats stats={data.stats} />
            
            <div className="space-y-3 pt-6 border-t border-slate-100">
                <a 
                    href={data.geojson_url}
                    download
                    className="flex items-center justify-center w-full gap-2 bg-slate-900 hover:bg-slate-800 text-white px-4 py-3 rounded-xl font-medium transition-colors"
                >
                    <Download size={18} />
                    Download GeoJSON
                </a>
                <button 
                    onClick={() => {
                        navigator.clipboard.writeText(window.location.href);
                        alert("Link copied!");
                    }}
                    className="flex items-center justify-center w-full gap-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 px-4 py-3 rounded-xl font-medium transition-colors"
                >
                    <Share2 size={18} />
                    Share Result
                </button>
            </div>
            
            <div className="pt-4 border-t border-slate-100 text-xs text-slate-400 font-mono text-center">
                Task ID: {taskId?.slice(0,8)}...<br/>
                Processing Time: {data.processing_time_seconds}s
            </div>
        </div>
      </div>

      {/* Main Map Content */}
      <div className="flex-1 relative bg-slate-200">
        <MapViewer mapUrl={data.map_url} />
      </div>
    </div>
  );
}

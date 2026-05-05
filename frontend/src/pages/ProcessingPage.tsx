import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { getStatus } from '../api/client';
import ProgressStepper from '../components/ProgressStepper';

export default function ProcessingPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const [countdown, setCountdown] = useState(60);

  const { data, isError } = useQuery({
    queryKey: ['status', taskId],
    queryFn: () => getStatus(taskId!).then(res => res.data),
    refetchInterval: (query) => {
        const state = query?.state?.data?.status;
        return (state === 'SUCCESS' || state === 'FAILURE') ? false : 2000;
    },
  });

  useEffect(() => {
    if (data?.status === 'SUCCESS') {
      navigate(`/result/${taskId}`);
    }
  }, [data?.status, navigate, taskId]);
  
  useEffect(() => {
    const timer = setInterval(() => setCountdown(c => Math.max(0, c - 1)), 1000);
    return () => clearInterval(timer);
  }, []);

  if (isError || data?.status === 'FAILURE') {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="bg-white p-8 rounded-2xl shadow-sm border border-red-200 text-center max-w-md">
          <div className="w-16 h-16 bg-red-100 text-red-600 border border-red-200 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold">!</div>
          <h2 className="text-xl font-bold text-slate-900 mb-2">Analysis Failed</h2>
          <p className="text-slate-500 mb-6 font-mono text-sm break-all">{data?.step || "An unknown error occurred during processing."}</p>
          <button 
            onClick={() => navigate('/')}
            className="bg-slate-900 hover:bg-slate-800 text-white px-6 py-2 rounded-lg font-medium w-full transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-slate-50">
      <div className="bg-white p-10 rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 max-w-lg w-full">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-slate-900">Processing Analysis</h2>
          <p className="text-slate-500 mt-2 font-medium">Estimated time remaining: {countdown}s</p>
        </div>
        
        <div className="bg-slate-50/50 p-6 rounded-2xl border border-slate-100">
            <ProgressStepper activeStep={data?.step} />
        </div>
      </div>
    </div>
  );
}

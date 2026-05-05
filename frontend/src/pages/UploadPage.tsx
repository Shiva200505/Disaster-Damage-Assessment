import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import DropZone from '../components/DropZone';
import { submitInference } from '../api/client';
import { Play } from 'lucide-react';

interface FormValues {
  disaster_name: string;
  map_threshold: number;
}

export default function UploadPage() {
  const navigate = useNavigate();
  const [preImage, setPreImage] = useState<File | null>(null);
  const [postImage, setPostImage] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { register, handleSubmit } = useForm<FormValues>({
    defaultValues: { disaster_name: "Tsunami Region A", map_threshold: 0.35 }
  });

  const onSubmit = async (data: FormValues) => {
    if (!preImage || !postImage) {
      setError("Please select both Pre and Post disaster images.");
      return;
    }
    setError(null);
    setIsSubmitting(true);
    
    try {
      const formData = new FormData();
      formData.append("pre_image", preImage);
      formData.append("post_image", postImage);
      formData.append("disaster_name", data.disaster_name);
      formData.append("map_threshold", data.map_threshold.toString());
      
      const res = await submitInference(formData);
      navigate(`/processing/${res.data.task_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Upload failed");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-12 max-w-4xl">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight">Disaster Damage Assessment</h1>
        <p className="text-slate-500 mt-2">Upload satellite imagery to automatically detect and classify destroyed buildings.</p>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-lg border border-red-200 text-sm">
            {error}
          </div>
        )}
        
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <DropZone label="Pre-Disaster Image" file={preImage} onFileSelect={setPreImage} />
            <DropZone label="Post-Disaster Image" file={postImage} onFileSelect={setPostImage} />
          </div>

          <div className="bg-slate-50 rounded-xl p-6 border border-slate-100">
            <h3 className="text-lg font-semibold text-slate-800 mb-4">Analysis Settings</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Disaster Name</label>
                <input 
                  type="text" 
                  {...register("disaster_name", { required: true })}
                  className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-sky-500 outline-none flex-1"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Sensitivity Threshold (0.1 - 0.9)</label>
                <div className="flex items-center gap-4">
                  <input 
                    type="range" 
                    min="0.1" max="0.9" step="0.05"
                    {...register("map_threshold")}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-sky-500 flex-1 my-auto"
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="flex justify-end">
            <button 
              type="submit" 
              disabled={isSubmitting}
              className="flex items-center gap-2 bg-sky-600 hover:bg-sky-700 text-white px-8 py-3 rounded-lg font-medium transition-colors disabled:opacity-50"
            >
              {isSubmitting ? "Uploading..." : "Start Analysis"}
              {!isSubmitting && <Play size={18} />}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

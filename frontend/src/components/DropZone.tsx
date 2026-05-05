import React, { useState } from 'react';
import { UploadCloud, X } from 'lucide-react';

interface DropZoneProps {
  label: string;
  file: File | null;
  onFileSelect: (file: File | null) => void;
}

export default function DropZone({ label, file, onFileSelect }: DropZoneProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const processFile = (f: File) => {
    if (f.size > 50 * 1024 * 1024) {
      alert("File is too large. Max 50MB.");
      return;
    }
    if (!['image/jpeg', 'image/png', 'image/tiff'].includes(f.type)) {
      alert("Only PNG, JPEG, and TIFF are supported.");
      return;
    }
    onFileSelect(f);
    const objectUrl = URL.createObjectURL(f);
    setPreview(objectUrl);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    onFileSelect(null);
    setPreview(null);
  };

  return (
    <div className="flex flex-col gap-2">
      <span className="font-semibold text-slate-700">{label}</span>
      <div 
        className={`relative flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
          isDragActive ? "border-sky-500 bg-sky-50" : "border-slate-300 bg-white hover:bg-slate-50"
        } ${preview ? "border-solid p-2" : ""}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => document.getElementById(`fileUpload-${label}`)?.click()}
      >
        <input 
          id={`fileUpload-${label}`} 
          type="file" 
          accept="image/png, image/jpeg, image/tiff"
          className="hidden" 
          onChange={handleChange} 
        />
        
        {preview ? (
          <div className="relative w-full h-48 rounded-lg overflow-hidden group">
            <img src={preview} alt={label} className="w-full h-full object-cover" />
            <div className="absolute inset-0 bg-black bg-opacity-40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
               <button 
                 onClick={clearFile}
                 className="p-2 bg-white rounded-full text-slate-800 hover:text-red-500"
               >
                 <X size={20} />
               </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center text-slate-500">
            <UploadCloud size={40} className="mb-3 text-slate-400" />
            <p className="text-sm font-medium">Drag & drop image here</p>
            <p className="text-xs text-slate-400 mt-1">PNG, JPEG, TIFF up to 50MB</p>
          </div>
        )}
      </div>
    </div>
  );
}

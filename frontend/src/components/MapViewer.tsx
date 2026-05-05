interface MapViewerProps {
  mapUrl: string;
}

export default function MapViewer({ mapUrl }: MapViewerProps) {
  return (
    <div className="w-full h-full bg-slate-100 relative group overflow-hidden">
      <iframe 
        src={mapUrl} 
        className="w-full h-full border-0 absolute inset-0"
        title="Damage Assessment Map"
      />
    </div>
  );
}

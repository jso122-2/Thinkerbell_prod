import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

export default function ArtifactDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center space-x-4 mb-6">
        <button
          onClick={() => navigate('/artifacts')}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Artifacts</span>
        </button>
      </div>
      
      <h1 className="text-4xl font-black text-black">
        Artifact <span className="tb-accent-green">Detail</span>
      </h1>
      
      <div className="tb-card">
        <h2 className="text-xl font-bold text-black mb-4">Artifact: {id}</h2>
        <p className="text-gray-600">Detailed artifact view coming soon...</p>
      </div>
    </div>
  );
}
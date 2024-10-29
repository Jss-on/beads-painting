// src/components/OptimizedSVGViewer.tsx
import React, { useState } from 'react';
import { Card, CardHeader, CardContent, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Download } from 'lucide-react';
import useIntersectionObserver from '@/lib/hooks/useIntersectionObserver';

interface OptimizedSVGViewerProps {
  filename: string;
  kernel: string;
  baseUrl: string;
}

const OptimizedSVGViewer: React.FC<OptimizedSVGViewerProps> = ({ 
  filename, 
  kernel, 
  baseUrl 
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const isVisible = useIntersectionObserver(`svg-container-${kernel}`);

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = `${baseUrl}/api/output/${filename}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle className="text-lg">{kernel}</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div 
          id={`svg-container-${kernel}`}
          className="relative aspect-w-16 aspect-h-12"
        >
          {isVisible && (
            <object
              data={`${baseUrl}/api/output/${filename}`}
              type="image/svg+xml"
              className="w-full h-full object-contain"
              onLoad={() => setIsLoading(false)}
            />
          )}
          {isLoading && isVisible && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            </div>
          )}
        </div>
        <div className="p-4">
          <Button
            variant="outline"
            className="w-full"
            onClick={handleDownload}
          >
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default OptimizedSVGViewer;
'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { X, Download, Upload, File } from 'lucide-react';
import OptimizedSVGViewer from '@/components/OptimizedSVGViewer';

const DitheringApp = () => {
  const [file, setFile] = useState<File | null>(null);
  const [fileSize, setFileSize] = useState('');
  const [progress, setProgress] = useState(0);
  const [processedImages, setProcessedImages] = useState<Record<string, string>>({});
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  const handleFileSelect = useCallback((selectedFile: File) => {
    if (selectedFile) {
      setFile(selectedFile);
      setFileSize(formatFileSize(selectedFile.size));
      setError(null);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      const droppedFile = e.dataTransfer.files[0];
      handleFileSelect(droppedFile);
    },
    [handleFileSelect]
  );

  const pollProgress = useCallback(async (sid: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/progress/${sid}`);
        const data = await response.json();

        if (data.error) {
          setError(data.error);
          setIsProcessing(false);
          clearInterval(interval);
          return;
        }

        setProgress(data.progress);

        if (data.complete) {
          setProcessedImages(data.images);
          setIsProcessing(false);
          clearInterval(interval);
        }
      } catch (error: any) {
        setError(error.message || 'Failed to fetch progress');
        setIsProcessing(false);
        clearInterval(interval);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const width = formRef.current?.width.value;
    const height = formRef.current?.height.value;
    
    if (width) formData.append('width', width);
    if (height) formData.append('height', height);

    setIsProcessing(true);
    setError(null);
    setProgress(0);
    setProcessedImages({});

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      pollProgress(data.session_id);
    } catch (error: any) {
      setError(error.message || 'Upload failed');
      setIsProcessing(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold text-center mb-12">Dithering Comparison Tool</h1>

      <div className="flex justify-center">
        <div className="w-full max-w-2xl">
          <Card>
            <CardHeader>
              <CardTitle>Add Image</CardTitle>
            </CardHeader>
            <CardContent>
              <form ref={formRef} onSubmit={handleSubmit}>
                {!file ? (
                  <div
                    className={`border-2 border-dashed rounded-lg p-8 mb-6 text-center cursor-pointer transition-colors
                      ${dragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-500 hover:bg-blue-50'}`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragOver(true);
                    }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                    />
                    <div className="flex flex-col items-center gap-4">
                      <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
                        <Upload className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <p className="text-lg font-medium">Upload your file here</p>
                        <p className="text-sm text-gray-500 mt-1">Files supported: JPG, PNG</p>
                      </div>
                      <p className="text-gray-400">OR</p>
                      <Button
                        type="button"
                        variant="outline"
                        onClick={(e) => {
                          e.stopPropagation();
                          fileInputRef.current?.click();
                        }}
                      >
                        BROWSE
                      </Button>
                      <p className="text-sm text-gray-500">Maximum size: 10MB</p>
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-50 rounded-lg p-4 flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      <File className="w-6 h-6 text-blue-600" />
                      <div>
                        <span className="font-medium">{file.name}</span>
                        <span className="text-sm text-gray-500 ml-2">{fileSize}</span>
                      </div>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      className="text-red-500 hover:text-red-700"
                      onClick={() => setFile(null)}
                    >
                      <X className="w-5 h-5" />
                    </Button>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Width:
                    </label>
                    <input
                      type="number"
                      name="width"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      min="1"
                      placeholder="Leave blank to keep ratio"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Height:
                    </label>
                    <input
                      type="number"
                      name="height"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      min="1"
                      placeholder="Leave blank to keep ratio"
                    />
                  </div>
                </div>

                <Button
                  type="submit"
                  className="w-full mt-6"
                  disabled={!file || isProcessing}
                >
                  {isProcessing ? 'Processing...' : 'Process Image'}
                </Button>
              </form>

              {isProcessing && (
                <div className="mt-4">
                  <div className="text-sm font-medium mb-2">Processing: {progress}%</div>
                  <Progress value={progress} />
                </div>
              )}

              {error && (
                <div className="mt-4 text-red-500">{error}</div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Results Grid */}
      {Object.keys(processedImages).length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 mt-12">
          {Object.entries(processedImages).map(([kernel, filename]) => (
            <OptimizedSVGViewer
            key={kernel}
            kernel={kernel}
            filename={filename}
            baseUrl="http://localhost:8000"
          />
          ))}
        </div>
      )}
    </div>
  );
};

export default DitheringApp;
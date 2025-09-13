import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Upload, FileText, Image, CheckCircle, AlertTriangle, Loader2 } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

interface UploadedFile {
  id: string;
  name: string;
  size: string;
  status: 'processing' | 'completed' | 'failed';
  fraudStatus?: 'clean' | 'suspicious' | 'fraudulent';
  extractedData?: any;
  error?: string;
}

export const ClaimUploader = () => {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const { toast } = useToast();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles.map(file => ({
      id: Math.random().toString(36).substring(7),
      name: file.name,
      size: `${(file.size / (1024 * 1024)).toFixed(2)} MB`,
      status: 'processing'
    }));

    setFiles(prev => [...prev, ...newFiles]);
    setUploadProgress(0);

    toast({
      title: "Upload Started",
      description: `Processing ${acceptedFiles.length} file(s) for fraud detection`,
    });

    // Process each file
    for (let i = 0; i < acceptedFiles.length; i++) {
      const file = acceptedFiles[i];
      const fileId = newFiles[i].id;

      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://127.0.0.1:8000/api/claims/upload', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        if (result.status === 'success') {
          setFiles(prev => prev.map(f => 
            f.id === fileId 
              ? { 
                  ...f, 
                  status: 'completed',
                  fraudStatus: result.extractedData.fraudStatus,
                  extractedData: result.extractedData
                }
              : f
          ));

          toast({
            title: "Processing Complete",
            description: `${file.name} analyzed - Status: ${result.extractedData.fraudStatus}`,
          });
        } else {
          throw new Error('Processing failed');
        }

      } catch (error) {
        console.error('Upload failed:', error);
        setFiles(prev => prev.map(f => 
          f.id === fileId 
            ? { 
                ...f, 
                status: 'failed',
                error: error instanceof Error ? error.message : 'Unknown error'
              } 
            : f
        ));

        toast({
          title: "Upload Failed",
          description: `Failed to process ${file.name}`,
          variant: "destructive",
        });
      }

      // Update progress
      setUploadProgress(((i + 1) / acceptedFiles.length) * 100);
    }
  }, [toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    multiple: true
  });

  const getStatusIcon = (status: string, fraudStatus?: string) => {
    if (status === 'processing') return <Loader2 className="h-4 w-4 animate-spin" />;
    if (status === 'failed') return <AlertTriangle className="h-4 w-4 text-danger" />;
    if (fraudStatus === 'fraudulent') return <AlertTriangle className="h-4 w-4 text-danger" />;
    if (fraudStatus === 'suspicious') return <AlertTriangle className="h-4 w-4 text-warning" />;
    return <CheckCircle className="h-4 w-4 text-success" />;
  };

  const getStatusBadge = (status: string, fraudStatus?: string) => {
    if (status === 'processing') return <Badge variant="secondary">Processing</Badge>;
    if (status === 'failed') return <Badge className="bg-danger text-danger-foreground">Failed</Badge>;
    if (fraudStatus === 'fraudulent') return <Badge className="bg-danger text-danger-foreground">Fraudulent</Badge>;
    if (fraudStatus === 'suspicious') return <Badge className="bg-warning text-warning-foreground">Suspicious</Badge>;
    return <Badge className="bg-success text-success-foreground">Clean</Badge>;
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200
          ${isDragActive 
            ? 'border-primary bg-primary/5 shadow-lg shadow-primary/20' 
            : 'border-border hover:border-primary/50 hover:bg-muted/30'
          }
        `}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center space-y-4">
          <div className="p-4 bg-primary/10 rounded-full">
            <Upload className="h-8 w-8 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground mb-2">
              {isDragActive ? "Drop files here" : "Upload Insurance Documents"}
            </h3>
            <p className="text-muted-foreground mb-4">
              Drag and drop PDF files or images, or click to browse
            </p>
            <Button variant="outline" className="border-primary/20 hover:bg-primary/5">
              Select Files
            </Button>
          </div>
        </div>
      </div>

      {/* Upload Progress */}
      {files.some(f => f.status === 'processing') && (
        <Card className="bg-muted/50">
          <CardHeader>
            <CardTitle className="text-sm">Processing Files...</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={uploadProgress} className="w-full" />
            <p className="text-xs text-muted-foreground mt-2">
              Extracting data and analyzing for fraud indicators
            </p>
          </CardContent>
        </Card>
      )}

      {/* Uploaded Files List */}
      {files.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold">Uploaded Files</h3>
          {files.map((file) => (
            <Card key={file.id} className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {file.name.endsWith('.pdf') ? (
                    <FileText className="h-8 w-8 text-danger" />
                  ) : (
                    <Image className="h-8 w-8 text-info" />
                  )}
                  <div>
                    <p className="font-medium text-foreground">{file.name}</p>
                    <p className="text-sm text-muted-foreground">{file.size}</p>
                    {file.error && (
                      <p className="text-xs text-danger mt-1">Error: {file.error}</p>
                    )}
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  {getStatusIcon(file.status, file.fraudStatus)}
                  {getStatusBadge(file.status, file.fraudStatus)}
                </div>
              </div>
              
              {file.extractedData && (
                <div className="mt-4 p-3 bg-muted/30 rounded-lg">
                  <h4 className="text-sm font-semibold mb-2">Extracted Information</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div><span className="font-medium">Claim ID:</span> {file.extractedData.claimId || 'N/A'}</div>
                    <div><span className="font-medium">Patient:</span> {file.extractedData.patientName || 'N/A'}</div>
                    <div><span className="font-medium">Hospital:</span> {file.extractedData.hospital || 'N/A'}</div>
                    <div><span className="font-medium">Amount:</span> ₹{file.extractedData.amount || 'N/A'}</div>
                    <div><span className="font-medium">Disease:</span> {file.extractedData.disease || 'N/A'}</div>
                    <div><span className="font-medium">Treatment:</span> {file.extractedData.treatment || 'N/A'}</div>
                  </div>
                  {file.extractedData.fraudReason && (
                    <div className="mt-2 p-2 bg-danger/10 rounded border-l-2 border-danger">
                      <p className="text-xs font-medium text-danger">Fraud Reason:</p>
                      <p className="text-xs text-danger">{file.extractedData.fraudReason}</p>
                    </div>
                  )}
                </div>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};
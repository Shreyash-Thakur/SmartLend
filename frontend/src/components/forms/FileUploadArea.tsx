import React, { useId, useState } from 'react'
import { Upload, FileText, Loader2 } from 'lucide-react'
import type { FileUploadProps } from '@/types/ui'

export const FileUploadArea: React.FC<FileUploadProps> = ({
  onFileSelect,
  acceptedFormats = ['pdf', 'csv'],
  maxSize = 5 * 1024 * 1024,
  maxSizeLabel = '5MB',
}) => {
  const inputId = useId()
  const [isDragging, setIsDragging] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [error, setError] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const validateFile = (file: File): boolean => {
    const fileExtension = file.name.split('.').pop()?.toLowerCase()
    
    if (!fileExtension || !acceptedFormats.includes(fileExtension)) {
      setError(`Only ${acceptedFormats.join(', ')} files are accepted`)
      return false
    }

    if (file.size > maxSize) {
      setError(`File size must be less than ${(maxSize / 1024 / 1024).toFixed(0)}MB`)
      return false
    }

    return true
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files && files[0]) {
      const file = files[0]
      if (validateFile(file)) {
        await handleFile(file)
      }
    }
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      if (validateFile(file)) {
        await handleFile(file)
      }
    }
  }

  const handleFile = async (file: File) => {
    setError('')
    setUploadedFile(file)
    
    if (onFileSelect) {
      setIsLoading(true)
      try {
        await onFileSelect(file)
      } catch (err) {
        setError(`Upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
      } finally {
        setIsLoading(false)
      }
    }
  }

  return (
    <div className="w-full">
      <div
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-lg p-8
          transition-all duration-200
          text-center cursor-pointer
          ${isDragging ? 'border-primary-500 bg-primary-50' : 'border-neutral-300 bg-neutral-50/80'}
          ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input
          type="file"
          accept={acceptedFormats.map((fmt) => `.${fmt}`).join(',')}
          onChange={handleFileChange}
          className="hidden"
          id={inputId}
          disabled={isLoading}
        />
        
        <label htmlFor={inputId} className="block cursor-pointer">
          <div className="flex flex-col items-center gap-2">
            {isLoading ? (
              <>
                <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
                <span className="text-sm text-neutral-600">Uploading...</span>
              </>
            ) : uploadedFile ? (
              <>
                <FileText className="h-8 w-8 text-primary-500" />
                <span className="text-sm font-medium text-neutral-900">{uploadedFile.name}</span>
                <span className="text-xs text-neutral-500">{(uploadedFile.size / 1024).toFixed(0)} KB</span>
              </>
            ) : (
              <>
                <Upload className="w-8 h-8 text-neutral-400 mx-auto" />
                <span className="text-sm font-medium text-neutral-900">Drag and drop supporting documents here</span>
                <span className="text-xs text-neutral-500">or click to browse files</span>
                <span className="text-xs text-neutral-400 mt-2">
                  Supported formats: {acceptedFormats.join(', ').toUpperCase()} up to {maxSizeLabel}
                </span>
              </>
            )}
          </div>
        </label>
      </div>

      {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
    </div>
  )
}

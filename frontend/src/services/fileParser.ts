export function getAcceptedFileTypes(formats: string[]) {
  return formats.map((format) => `.${format}`).join(',')
}

export async function simulateFileExtraction(file: File) {
  return {
    fileName: file.name,
    documentType: file.name.split('.').pop() ?? 'file',
    uploadedAt: new Date().toISOString(),
    fileSize: file.size,
    extractedData: {
      preview: 'Document queued for backend parsing.',
    },
  }
}

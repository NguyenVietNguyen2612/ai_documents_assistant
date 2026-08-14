import { useState } from "react"

function DocumentUpload({ onUploadSuccess }) {
    const [file, setFile] = useState(null)
    const [uploading, setUploading] = useState(false)

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0]

        if (!selectedFile) {
            return
        }

        setFile(selectedFile)
    }

    const handleUpload = async () => {
        if (!file) {
            return
        }

        setUploading(true)

        const formData = new FormData()
        formData.append("file", file)

        try {
            const response = await fetch("http://localhost:8000/documents/upload", {
                method: "POST",
                body: formData,
            })

            if (response.ok) {
                setFile(null)
                if (onUploadSuccess) {
                    onUploadSuccess()
                }
            } else {
                console.error("Upload failed")
            }
        } catch (error) {
            console.error("Error uploading file:", error)
        } finally {
            setUploading(false)
        }
    }

    return (
        <div className="upload-section">
            <label className="upload-box">
                <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    hidden
                />

                <div className="upload-icon">
                    ↑
                </div>

                <div className="upload-title">
                    Upload document
                </div>

                <div className="upload-description">
                    Click to browse your files
                </div>

                <div className="upload-format">
                    PDF files only · Max 20 MB
                </div>
            </label>

            {file && (
                <div className="selected-file">
                    <div className="file-icon">PDF</div>

                    <div className="file-info">
                        <span className="file-name">
                            {file.name}
                        </span>

                        <span className="file-size">
                            {(file.size / 1024 / 1024).toFixed(2)} MB
                        </span>
                    </div>

                    <button
                        className="upload-button"
                        onClick={handleUpload}
                        disabled={uploading}
                    >
                        {uploading ? "Uploading..." : "Upload"}
                    </button>
                </div>
            )}
        </div>
    )
}

export default DocumentUpload
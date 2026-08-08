import { useState } from "react"

function DocumentUpload() {
    const [file, setFile] = useState(null)

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0]

        if (!selectedFile) {
            return
        }

        setFile(selectedFile)
    }

    const handleUpload = () => {
        if (!file) {
            console.log("No file selected")
            return
        }

        console.log("Selected file:", file)
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
                    >
                        Upload
                    </button>
                </div>
            )}
        </div>
    )
}

export default DocumentUpload
import React from "react"

/**
 * Render Markdown chuẩn tích hợp Interactive Citations:
 * - Chuyển đổi [Tài liệu: <tên_file> - Trang <số_trang>] thành Citation Badge có thể nhấp chuột
 * - Hỗ trợ Bảng biểu Markdown (| col1 | col2 |)
 * - Hỗ trợ Tiêu đề (#, ##, ###), Danh sách (- , 1.), In đậm (**), In nghiêng (*), Khối code
 */
function MarkdownViewer({ content = "", onCitationClick }) {
    if (!content) return null

    // Hàm render đoạn inline (xử lý bold, italic, code, math, và citation badges)
    const renderInline = (text) => {
        if (!text) return null

        // Pattern cho citations: [Tài liệu: CS338.pdf - Trang 2] hoặc [Source 1: CS338.pdf - Trang 2]
        const citationRegex = /\[(?:Source\s*\d+:\s*)?Tài liệu:\s*([^\]\-]+?)(?:\s*-\s*Trang\s*(\d+))?\]|\[Source\s*(\d+)\]/gi

        // Tokenize theo citation
        const parts = []
        let lastIndex = 0
        let match

        while ((match = citationRegex.exec(text)) !== null) {
            // Phần text trước match
            if (match.index > lastIndex) {
                parts.push({ type: "text", value: text.substring(lastIndex, match.index) })
            }

            const rawMatch = match[0]
            const docName = (match[1] || "").trim()
            const pageNum = match[2] ? parseInt(match[2], 10) : (match[3] ? parseInt(match[3], 10) : null)

            parts.push({
                type: "citation",
                raw: rawMatch,
                docName: docName || (match[3] ? `Nguồn ${match[3]}` : "Tài liệu"),
                page: pageNum,
            })

            lastIndex = citationRegex.lastIndex
        }

        if (lastIndex < text.length) {
            parts.push({ type: "text", value: text.substring(lastIndex) })
        }

        // Render từng part
        return parts.map((part, pIdx) => {
            if (part.type === "citation") {
                return (
                    <button
                        key={pIdx}
                        type="button"
                        className="interactive-citation-badge"
                        onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            if (onCitationClick) {
                                onCitationClick({ documentName: part.docName, page: part.page })
                            }
                        }}
                        title={`Nhấp để mở tài liệu ${part.docName} (Trang ${part.page || 1})`}
                    >
                        <span className="citation-icon">📄</span>
                        <span className="citation-doc-name">{part.docName}</span>
                        {part.page && (
                            <span className="citation-page-badge">Trang {part.page}</span>
                        )}
                    </button>
                )
            }

            // Với text thường: xử lý tiếp bold, code, italic, math
            return renderSimpleInline(part.value, pIdx)
        })
    }

    // Format bold (**), code (`), italic (*), math ($)
    const renderSimpleInline = (textStr, keyPrefix) => {
        // Regex bóc tách `inline code`, **bold**, *italic*, $math$
        const inlineRegex = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\$[^$]+\$)/g
        const tokens = textStr.split(inlineRegex)

        return (
            <React.Fragment key={keyPrefix}>
                {tokens.map((tok, tIdx) => {
                    if (!tok) return null
                    if (tok.startsWith("`") && tok.endsWith("`") && tok.length >= 2) {
                        return <code key={tIdx} className="md-inline-code">{tok.slice(1, -1)}</code>
                    }
                    if (tok.startsWith("**") && tok.endsWith("**") && tok.length >= 4) {
                        return <strong key={tIdx} className="md-bold">{tok.slice(2, -2)}</strong>
                    }
                    if (tok.startsWith("*") && tok.endsWith("*") && tok.length >= 2) {
                        return <em key={tIdx} className="md-italic">{tok.slice(1, -1)}</em>
                    }
                    if (tok.startsWith("$") && tok.endsWith("$") && tok.length >= 2) {
                        return <span key={tIdx} className="md-math">{tok.slice(1, -1)}</span>
                    }
                    return tok
                })}
            </React.Fragment>
        )
    }

    // Phân tích các khối dòng (Blocks: headers, tables, lists, codeblocks, paragraphs)
    const lines = content.split("\n")
    const blocks = []
    let inCodeBlock = false
    let codeBlockLang = ""
    let codeBlockLines = []
    let inTable = false
    let tableRows = []

    const flushTable = () => {
        if (tableRows.length > 0) {
            blocks.push({ type: "table", rows: [...tableRows] })
            tableRows = []
            inTable = false
        }
    }

    const flushCodeBlock = () => {
        if (codeBlockLines.length > 0 || inCodeBlock) {
            blocks.push({
                type: "codeblock",
                lang: codeBlockLang,
                content: codeBlockLines.join("\n"),
            })
            codeBlockLines = []
            inCodeBlock = false
            codeBlockLang = ""
        }
    }

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i]
        const trimmed = line.trim()

        // 1. Khối Code Block ```
        if (trimmed.startsWith("```")) {
            if (inTable) flushTable()
            if (inCodeBlock) {
                flushCodeBlock()
            } else {
                inCodeBlock = true
                codeBlockLang = trimmed.slice(3).trim()
                codeBlockLines = []
            }
            continue
        }

        if (inCodeBlock) {
            codeBlockLines.push(line)
            continue
        }

        // 2. Khối Table Markdown (| ... |)
        if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
            // Kiểm tra dòng separator |---|---|
            const isSeparator = /^\|(?:\s*:?-+:?\s*\|)+$/.test(trimmed)
            if (isSeparator) {
                // Đánh dấu dòng header trước đó
                inTable = true
                continue
            }
            const cells = trimmed
                .slice(1, -1)
                .split("|")
                .map((c) => c.trim())
            tableRows.push(cells)
            inTable = true
            continue
        } else if (inTable) {
            flushTable()
        }

        // 3. Tiêu đề (#, ##, ###)
        if (trimmed.startsWith("### ")) {
            blocks.push({ type: "h3", text: trimmed.slice(4) })
            continue
        }
        if (trimmed.startsWith("## ")) {
            blocks.push({ type: "h2", text: trimmed.slice(3) })
            continue
        }
        if (trimmed.startsWith("# ")) {
            blocks.push({ type: "h1", text: trimmed.slice(2) })
            continue
        }

        // 4. Danh sách gạch đầu dòng (- hoặc *)
        if (/^[-*]\s+/.test(trimmed)) {
            blocks.push({ type: "ul-item", text: trimmed.replace(/^[-*]\s+/, "") })
            continue
        }

        // 5. Danh sách số (1. , 2. )
        if (/^\d+\.\s+/.test(trimmed)) {
            const num = trimmed.match(/^(\d+)\.\s+/)[1]
            blocks.push({ type: "ol-item", num, text: trimmed.replace(/^\d+\.\s+/, "") })
            continue
        }

        // 6. Blockquote (> )
        if (trimmed.startsWith("> ")) {
            blocks.push({ type: "blockquote", text: trimmed.slice(2) })
            continue
        }

        // 7. Dòng trống
        if (!trimmed) {
            blocks.push({ type: "empty" })
            continue
        }

        // 8. Đoạn văn thường (paragraph)
        blocks.push({ type: "p", text: line })
    }

    if (inCodeBlock) flushCodeBlock()
    if (inTable) flushTable()

    return (
        <div className="md-content-body">
            {blocks.map((block, bIdx) => {
                switch (block.type) {
                    case "h1":
                        return <h2 key={bIdx} className="md-h1">{renderInline(block.text)}</h2>
                    case "h2":
                        return <h3 key={bIdx} className="md-h2">{renderInline(block.text)}</h3>
                    case "h3":
                        return <h4 key={bIdx} className="md-h3">{renderInline(block.text)}</h4>
                    case "blockquote":
                        return (
                            <blockquote key={bIdx} className="md-blockquote">
                                {renderInline(block.text)}
                            </blockquote>
                        )
                    case "ul-item":
                        return (
                            <div key={bIdx} className="md-list-item md-ul-item">
                                <span className="md-bullet">•</span>
                                <span className="md-list-text">{renderInline(block.text)}</span>
                            </div>
                        )
                    case "ol-item":
                        return (
                            <div key={bIdx} className="md-list-item md-ol-item">
                                <span className="md-number">{block.num}.</span>
                                <span className="md-list-text">{renderInline(block.text)}</span>
                            </div>
                        )
                    case "codeblock":
                        return (
                            <div key={bIdx} className="md-codeblock-container">
                                {block.lang && <div className="md-codeblock-lang">{block.lang}</div>}
                                <pre className="md-codeblock">
                                    <code>{block.content}</code>
                                </pre>
                            </div>
                        )
                    case "table":
                        if (!block.rows || block.rows.length === 0) return null
                        const headerRow = block.rows[0]
                        const bodyRows = block.rows.slice(1)
                        return (
                            <div key={bIdx} className="md-table-wrapper">
                                <table className="md-table">
                                    <thead>
                                        <tr>
                                            {headerRow.map((cell, cIdx) => (
                                                <th key={cIdx}>{renderInline(cell)}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {bodyRows.map((row, rIdx) => (
                                            <tr key={rIdx}>
                                                {row.map((cell, cIdx) => (
                                                    <td key={cIdx}>{renderInline(cell)}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )
                    case "empty":
                        return <div key={bIdx} className="md-spacing" />
                    case "p":
                    default:
                        return (
                            <p key={bIdx} className="md-p">
                                {renderInline(block.text)}
                            </p>
                        )
                }
            })}
        </div>
    )
}

export default MarkdownViewer


def create_chunks(pages, chunk_size=500, chunk_overlap=100):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    separators = ["\n\n", "\n", ". ", " ", ""]

    def recursive_split(text, separator_index=0):

        text = text.strip()

        if not text:
            return []

        if len(text) <= chunk_size:
            return [text]

        if separator_index >= len(separators):
            raise ValueError("Unable to split text safely")

        separator = separators[separator_index]

        if not separator:
            # Character fallback
            step = chunk_size - chunk_overlap

            return [
                text[i:i + chunk_size]
                for i in range(0, len(text), step)
            ]

        parts = text.split(separator)

        if len(parts) == 1:
            return recursive_split(text, separator_index + 1)

        chunks = []
        current = []

        for part in parts:

            if not part:
                continue

            if len(part) > chunk_size:

                if current:
                    chunks.append(separator.join(current))
                    current = []

                chunks.extend(
                    recursive_split(part, separator_index + 1)
                )
                continue

            candidate = separator.join(current + [part])

            if len(candidate) <= chunk_size:
                current.append(part)

            else:

                if current:
                    chunks.append(separator.join(current))

                current = [part]

        if current:
            chunks.append(separator.join(current))

        # Recursively split any remaining oversized chunks
        final_chunks = []

        for chunk in chunks:

            if len(chunk) > chunk_size:
                final_chunks.extend(
                    recursive_split(chunk, separator_index + 1)
                )
            else:
                final_chunks.append(chunk)

        return final_chunks

    def add_overlap(chunks):

        if chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):

            previous = chunks[i - 1]
            current = chunks[i]

            # Preserve the complete current chunk.
            # Add overlap only when it fits.
            overlap = previous[-chunk_overlap:]

            if len(overlap) + len(current) <= chunk_size:
                result.append(overlap + current)
            else:
                result.append(current)

        return result

    all_chunks = []

    for page in pages:

        chunks = recursive_split(page["content"])

        chunks = add_overlap(chunks)

        for chunk in chunks:

            all_chunks.append({
                "content": chunk,
                "page_number": page["page_number"]
            })

    return all_chunks

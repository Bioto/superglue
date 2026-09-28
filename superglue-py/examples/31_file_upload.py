"""Example 31: inline file attachment + optional upload."""

import os
import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "openai:gpt-4o-mini"),
)

sample = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 200 200]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 24 Tf 20 100 Td (Hi) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000060 00000 n 
0000000114 00000 n 
0000000220 00000 n 
0000000314 00000 n 
trailer<</Size 6/Root 1 0 R>>
startxref
380
%%EOF
"""
msg = superglue.Client.message_with_file_bytes(
    "sample.pdf",
    sample,
    "Summarize this file in one sentence.",
)
out = client.complete_messages([msg])
print("inline:", out.content)

upload_path = os.environ.get("SUPERGLUE_UPLOAD_PATH")
if upload_path:
    uploaded = client.upload_file(upload_path, purpose="user_data")
    print("uploaded file_id:", uploaded.file_id)

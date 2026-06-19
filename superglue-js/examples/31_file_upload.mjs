/**
 * Example 31: inline file bytes + optional upload path.
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient, messageWithFileBytes } = require("../dist/facade.js");

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error("Set OPENAI_API_KEY to run this example.");
  process.exit(1);
}

const client = createClient({
  apiKey,
  model: process.env.OPENAI_MODEL ?? "openai:gpt-4o-mini",
});

const sample = Buffer.from(
  `%PDF-1.4
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
`,
);
const msg = messageWithFileBytes(
  "sample.pdf",
  sample,
  "Summarize this file in one sentence.",
);
const out = await client.completeMessages([msg]);
console.log("inline:", out.content);

const uploadPath = process.env.SUPERGLUE_UPLOAD_PATH;
if (uploadPath) {
  const uploaded = await client.uploadFile(uploadPath, "user_data");
  console.log("uploaded file_id:", uploaded.fileId);
}

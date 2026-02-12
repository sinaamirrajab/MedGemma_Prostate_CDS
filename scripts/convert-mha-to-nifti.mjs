import fs from "node:fs";
import path from "node:path";
import { readImageNode, writeImageNode } from "@itk-wasm/image-io";

const conversions = [
  { input: "imgs/t2w.mha", output: "public/imgs/t2w.nii.gz" },
  { input: "imgs/adc.mha", output: "public/imgs/adc.nii.gz" },
];

const ensureDir = (filePath) => {
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });
};

const convertOne = async ({ input, output }) => {
  if (!fs.existsSync(input)) {
    throw new Error(`Missing input file: ${input}`);
  }
  ensureDir(output);
  console.log(`Converting ${input} -> ${output}`);
  const image = await readImageNode(input);
  await writeImageNode(image, output);
};

const run = async () => {
  for (const item of conversions) {
    await convertOne(item);
  }
  console.log("Done.");
};

run().catch((error) => {
  console.error(error);
  process.exit(1);
});

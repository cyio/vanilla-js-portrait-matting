import { env, pipeline } from "@xenova/transformers"
// Specify a custom location for models (defaults to '/models/').
env.localModelPath = './';

// Disable the loading of remote models from the Hugging Face Hub:
env.allowRemoteModels = false;

const modelPath = "modnet.onnx";

async function main() {
  // 创建管道
  const task = await pipeline("image-to-image", modelPath);

  // 进行预测
  const output = await task(imagePath);

  // 保存结果
  await writeFile(outputPath, output.matte);
}

main();

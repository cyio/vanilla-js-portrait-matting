import { env, AutoProcessor, pipeline } from "@xenova/transformers";

env.localModelPath = './';
env.allowRemoteModels = false;

const model = await AutoProcessor.from_pretrained("modnet");
// const model = await pipeline('image-to-image', 'modnet');

// 使用模型进行预测

const output = await model({
  image: {
    path: "./sample.jpeg",
  },
});

// 保存输出

await output.save("./output/");

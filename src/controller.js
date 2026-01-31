import { predictImage } from './predict-service.js';
 
export const getPredictResult = async (req, res) => {
  const photo = req.files[0].buffer;
  const predict = await predictImage(photo);
  const { diseaseLabel, confidenceScore } = predict;
 
  return res.status(200).json({
    status: 'success',
    message: 'Predict success',
    data: {
      disease: diseaseLabel,
      confidenceScore
    }
  });
};
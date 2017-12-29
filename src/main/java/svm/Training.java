package svm;

import net.sf.javaml.utils.ArrayUtils;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import java.io.File;

public class Training {

    private static final int FEATURE_DIMENSIONS = 59;

    public static double[] getLabels(int dataCounter){
        double[] labels = new double[dataCounter];
        for (int i = 0; i < labels.length; i++){
            labels[i] = 0;
        }
        return labels;
    }

    public static double[][] getTrainingData()
    {
        File dir = new File("training/images");
        int numberOfData = dir.listFiles().length;
        double[][] training_data = new double[numberOfData][FEATURE_DIMENSIONS];

        int fileCounter = 0;
        for (File img : dir.listFiles())
        {
            int featureCounter = 0;
            Mat src = Imgcodecs.imread(img.getAbsolutePath());
            for (int feature : URLBP.getURLBFeatures(src)){
                training_data[fileCounter][featureCounter++] = (double) feature;
            }
            // use color values as features instead
//            for (double feature : getColorFeatures(src)){
//                training_data[fileCounter][featureCounter++] = feature;
//            }
            fileCounter++;
        }
        for (double[] data : training_data) {
            ArrayUtils.normalize(data);
        }
        return training_data;
    }

    public static double[] getColorFeatures(Mat image){
        double[] data = new double[image.rows() * image.cols()];
        int counter = 0;
        for (int i = 0; i < image.rows(); i++){
            for (int j = 0; j < image.cols(); j++){
                data[counter++] = image.get(i,j)[0];
            }
        }
        return data;
    }
}

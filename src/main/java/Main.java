import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import hep.aida.ref.Test;
import net.sf.javaml.utils.ArrayUtils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import svm.SVM;
import svm.Training;
import svm.URLBP;

import java.io.File;
import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        System.load("src\\main\\resources\\natives\\opencv_320_64.dll");
        double[][] train = Training.getTrainingData(); // load training images
        SVM.setParameters();
        SVM.createProblem(train,Training.getLabels(train.length));
        SVM.save("svm.model",SVM.train()); // train and save svm model
        File dir = new File("testing/images"); // load testing images
        for (File img : dir.listFiles()){
            // preprocess image
            Mat test = Imgcodecs.imread(img.getAbsolutePath());
            Imgproc.resize(test,test,new Size(100,50));
            int[] data = URLBP.getURLBFeatures(test);
            double[] dData = Doubles.toArray(Ints.asList(data));
            ArrayUtils.normalize(dData);
            System.out.println(img.getName());
            // evaluate/test against model
            SVM.evaluate(dData,SVM.loadModel());
            System.out.println();
        }
    }
}

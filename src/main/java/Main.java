import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import hep.aida.ref.Test;
import libsvm.svm_model;
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
import java.util.TreeMap;

public class Main {

    public static void main(String[] args) throws IOException {
        System.out.println("extracting features");
        System.load("C:\\Users\\310297685\\IdeaProjects\\libsvm\\src\\main\\resources\\natives\\opencv_320_64.dll");
//        double[][] train = Training.getTrainingData("C:\\Users\\310297685\\Desktop\\training1\\text"); // load training images
//        SVM.setParameters();
//        System.out.println("training started");
//        SVM.createProblem(train,Training.getLabels(train.length));
//
//        SVM.save("C:\\Users\\310297685\\IdeaProjects\\libsvm\\src\\main\\resources\\models\\",SVM.train()); // train and save svm model
//        System.out.println("model saved");


        File dir = new File("C:\\Users\\310297685\\Desktop\\deep_test_temp\\testing\\non_text"); // load testing images
        svm_model svm1500r = SVM.loadModel("C:\\Users\\310297685\\IdeaProjects\\libsvm\\src\\main\\resources\\models\\svm1500r.model");
        svm_model svm2000r = SVM.loadModel("C:\\Users\\310297685\\IdeaProjects\\libsvm\\src\\main\\resources\\models\\svm2000r.model");
        svm_model svm2000 = SVM.loadModel("C:\\Users\\310297685\\IdeaProjects\\libsvm\\src\\main\\resources\\models\\svm2000.model");
        int positives1 = 0;
        int negatives1 = 0;
        int positives2 = 0;
        int negatives2 = 0;
        int positives3 = 0;
        int negatives3 = 0;
        for (File img : dir.listFiles()){
            Mat test = Imgcodecs.imread(img.getAbsolutePath());
            //System.out.print(img.getName());
            for (Mat subregion : Training.getSubregions(test)){
                int[] intData = URLBP.get_URLBP_Features(subregion);
                double[] doubleData = Training.intArrayToDouble(intData);
                Training.normalizeArray(doubleData);
                double score1 = SVM.evaluate(doubleData,svm1500r);
                double score2 = SVM.evaluate(doubleData,svm2000r);
                double score3 = SVM.evaluate(doubleData,svm2000);
                if (Double.compare(score1,1.0) == 0) positives1++;
                if (Double.compare(score1,-1.0) == 0) negatives1++;
                if (Double.compare(score2,1.0) == 0) positives2++;
                if (Double.compare(score2,-1.0) == 0) negatives2++;
                if (Double.compare(score3,1.0) == 0) positives3++;
                if (Double.compare(score3,-1.0) == 0) negatives3++;
            }
        }
        System.out.println(positives1+"\t"+negatives1+"\t"+positives2+"\t"+negatives2);
    }
}

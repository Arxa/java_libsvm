package svm;

import libsvm.*;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.SparseInstance;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * Support Vector Machine
 */
public class SVM {

    private static svm_problem prob = new svm_problem();
    private static svm_parameter param = new svm_parameter();
    private static final int FEATURE_DIMENTIONS = 59;

    public static void setParameters()
    {
        param.svm_type = svm_parameter.ONE_CLASS;
        param.nu = 0.01;
        param.kernel_type = svm_parameter.LINEAR;
        param.probability = 1;
        //param.cache_size = 2000;
    }

    public static void createProblem(double[][] train, double[] labels)
    {
        int dataCount = train.length;
        //prob.y = new double[dataCount];
        prob.l = dataCount;
        prob.x = new svm_node[dataCount][];
        for (int i = 0; i < dataCount; i++)
        {
            double[] features = train[i];
            prob.x[i] = new svm_node[features.length];
            for (int j = 0; j < features.length; j++)
            {
                svm_node node = new svm_node();
                node.index = j;
                node.value = features[j];
                prob.x[i][j] = node;
            }
        }
        prob.y = Arrays.copyOf(labels,labels.length); // check this.
    }

    public static svm_model train() {
        System.out.println(svm.svm_check_parameter(prob,param));
        return svm.svm_train(prob, param); // returns the model
    }

    public static void save(String path,svm_model model)
    {
        try {
            svm.svm_save_model(path, model);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static svm_model loadModel(String path) throws IOException
    {
        return svm.svm_load_model(path);
    }

    public static double evaluate(double[] features, svm_model model)
    {
        svm_node[] nodes = new svm_node[features.length];
        for (int i = 0; i < features.length; i++)
        {
            svm_node node = new svm_node();
            node.index = i;
            node.value = features[i];
            nodes[i] = node;
        }
        return svm.svm_predict(model,nodes);
//        int totalClasses = 2;
//        int[] labels = new int[totalClasses];
//        svm.svm_get_labels(model,labels);
//
//        double[] prob_estimates = new double[totalClasses];
//        double v = svm.svm_predict_probability(model, nodes, prob_estimates);
//
//        //System.out.println("SVM: "+v);
//        return v;

    }
}

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

//    public static svm_parameter gridSearch(double[][] train, svm_parameter param)
//    {
//        Dataset dataset = new DefaultDataset();
//        for (int i = 0; i < Training.getNumber_of_text_data() * Training.SUBREGIONS; i++)
//        {
//            Instance instance = new SparseInstance();
//            for (int y = 0; y < train[i].length; y++)
//            {
//                instance.put(y, train[i][y]);
//            }
//            instance.setClassValue("text");
//            dataset.add(instance);
//        }
//        for (int i = Training.getNumber_of_text_data() * Training.SUBREGIONS; i < train.length; i++)
//        {
//            Instance instance = new SparseInstance();
//            for (int y = 0; y < train[i].length; y++)
//            {
//                instance.put(y, train[i][y]);
//            }
//            instance.setClassValue("background");
//            dataset.add(instance);
//        }
//        double[] C = new double[11];
//        double[] Y = new double[11];
//        for (int i = 0; i <= 10; i++) {
//            C[i] = Math.pow(2.0,(double)i);
//        }
//        int v = 11;
//        for (int i = 0; i <= 10; i++) {
//            Y[i] = 1.0 / Math.pow(2.0,(double)v--);
//        }
//        //Instances data = new Instances("myData",vector,3000);
//        LibSVM libsvm = new LibSVM();
//        //libsvm.buildClassifier(dataset);
//        //libsvm.setParameters(param);
//        GridSearch gr = new GridSearch(libsvm,dataset,10);
//        System.out.println("ok");
//        return gr.search(libsvm.getParameters(),C,Y);
//    }


    public static void setParameters()
    {
        //param.C = 1024; Used for multiple clasiffication
        param.svm_type = svm_parameter.ONE_CLASS;
        param.nu = 0.01;
        param.kernel_type = svm_parameter.LINEAR;
        param.probability = 1;
        //param.gamma = 0.5; // Y No need if using LINEAR kernel
    }

//    public static void testGridSearch() throws Exception
//    {
//        Files.write(Paths.get("svm_stats.txt"),"\n".getBytes());
//
//        double[] C = new double[11];
//        double[] Y = new double[11];
//
//        // C: 2^0...2^10
//        for (int i = 0; i <= 10; i++) {
//            C[i] = Math.pow(2.0,(double)i);
//        }
//
//        // Y: 2^-11...2^-1
//        int v = 11;
//        for (int i = 0; i <= 10; i++) {
//            Y[i] = 1.0 / Math.pow(2.0,(double)v--);
//        }
//
//        for (int c=0; c <= 10; c++ )
//        {
//            for (int y = 0; y <= 10; y++)
//            {
//
//                final int finalC = (int)C[c];
//                final double finalY = Y[y];
//
//                //final String accurasy;
//                ExecutorService executor = Executors.newSingleThreadExecutor();
//                Future<String> future = executor.submit(new Callable<String>() {
//                    @Override
//                    public String call() throws Exception {
//                        return gridSearch_test(finalC,finalY);
//                    }
//                });
//                try {
//                    System.out.println("Started..");
//                    Files.write(Paths.get("svm_stats.txt"),("Calculating For C:"+(int)C[c]+", Y:"+Y[y]+"\n").
//                            getBytes(), StandardOpenOption.APPEND);
//                    Files.write(Paths.get("svm_stats.txt"),("For C:"+(int)C[c]+", Y:"+Y[y]+" Accuracy: "+
//                            future.get(240, TimeUnit.SECONDS)+"\n\n").getBytes(),StandardOpenOption.APPEND);
//
//                    //System.out.println("For C:"+c+", Y:"+y+" Accuracy: "+future.get(30, TimeUnit.SECONDS));
//                    //System.out.println("Finished!");
//                } catch (TimeoutException e) {
//                    future.cancel(true);
//                    Files.write(Paths.get("svm_stats.txt"),"Terminated!\n\n".getBytes(),StandardOpenOption.APPEND);
//                }
//                executor.shutdownNow();
//            }
//        }
//    }



    public static void gridSearch()
    {
        double[] target = new double[prob.y.length];
        double[] C = new double[11];
        double[] Y = new double[11];

        // C: 2^0...2^10
        for (int i = 0; i <= 10; i++) {
            C[i] = Math.pow(2.0,(double)i);
        }

        // Y: 2^-11...2^-1
        int v = 11;
        for (int i = 0; i <= 10; i++) {
            Y[i] = 1.0 / Math.pow(2.0,(double)v--);
        }

        double bestC = 0, bestY = 0;
        int accuracy = 0;
        for (int c=0; c <= 10; c++ )
        {
            for (int y=0; y <= 10; y++ )
            {
                System.out.printf("Calculating c="+c+", y="+y);
                param.C = C[c];
                param.gamma = Y[y];
                svm.svm_cross_validation(prob,param,3,target);
                int correctCounter = 0;
                for (int k=0; k < target.length; k++)
                {
                    if (Double.compare(target[k],prob.y[k]) == 0) {
                        correctCounter++;
                    }
                }
                if (correctCounter > accuracy)
                {
                    accuracy = correctCounter;
                    bestC = C[c];
                    bestY = Y[y];
                }
            }
        }
        System.out.println("Best C: "+bestC+" Best Y: "+bestY+" Accuracy: "+accuracy);
    }

//    public static String gridSearch_test(int c, double y)
//    {
//        boolean flag = false;
//        double[] target = new double[prob.y.length];
//
//        double bestC = 0, bestY = 0;
//        int accuracy = 0;
//        param.C = c;
//        param.gamma = y;
//        svm.svm_cross_validation(prob,param,3,target);
//        int correctCounter = 0;
//        for (int k=0; k < target.length; k++)
//        {
//            if (Double.compare(target[k],prob.y[k]) == 0) {
//                correctCounter++;
//            }
//        }
//        if (correctCounter > accuracy) {
//            accuracy = correctCounter;
//        }
//        flag = true;
//        if (flag){
//            return String.valueOf(accuracy);
//        }
//        else return "-1";
//    }

    public static void createProblem(double[][] train, double[] labels)
    {
        int dataCount = train.length;
        prob.y = new double[dataCount];
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
        prob.y = Arrays.copyOf(labels,dataCount); // check this.
    }

    public static svm_model train() {
        return svm.svm_train(prob, param); // returns the model
    }

    public static void save(String filename,svm_model model)
    {
        try {
            svm.svm_save_model("C:\\Users\\310297685\\IdeaProjects\\Thesis\\VideoText_Extractor\\src\\main\\resources\\models\\"+filename, model);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static svm_model loadModel() throws IOException
    {
        return svm.svm_load_model("C:\\Users\\310297685\\IdeaProjects\\Thesis\\VideoText_Extractor\\src\\main\\resources\\models\\svm_model_big_urlbp_abs.model");
    }

    public static double evaluate(double[] features, svm_model model)
    {
        svm_node[] nodes = new svm_node[features.length-1];
        for (int i = 1; i < features.length; i++)
        {
            svm_node node = new svm_node();
            node.index = i;
            node.value = features[i];

            nodes[i-1] = node;
        }

        int totalClasses = 2;
        int[] labels = new int[totalClasses];
        svm.svm_get_labels(model,labels);

        double[] prob_estimates = new double[totalClasses];
        double v = svm.svm_predict_probability(model, nodes, prob_estimates);

        System.out.println("SVM: "+v);
        return v;

    }

    public static svm_parameter getParam() {
        return param;
    }


}

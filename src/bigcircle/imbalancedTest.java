package bigcircle;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class imbalancedTest{
	
	public static void main(String[] args) throws Exception{
		int m_NumberIterations = 50;
		int classifierType = 0;         //0:j48,  1:ANN,  2:SMO with RBFKernel
		int nr_fold = 5;
		int extra_fold = 10;
	    double minorityLabel = 0;
	    
	    String path = "./FinalResult/V22+maj0.3"; 
		File root = new File("./arff-new");
		
		
		File[] files = root.listFiles();
		
		double[] rFmeasure = new double[files.length];
	    double[] rAuc = new double[files.length];
	    double[] rAucStd = new double[files.length];
	    double[] rRecall = new double[files.length];
	    double[] rAcc = new double[files.length];
	    double[] rPrecision = new double[files.length];
	    double[] rGmean = new double[files.length];
	    double[] rGmeanStd = new double[files.length];
	    
	    String[] fileNames = new String[files.length];
	   
	    
	    Random random = new Random();
	    
	    
	    File dir = new File(path);
	    if (dir.exists()==false || dir.isDirectory()==false) {
			dir.mkdirs();
		}
	    
		for(int f=0;f<files.length;f++){
			int pos = files[f].getName().lastIndexOf(".arff");
		    String fileName = files[f].getName().substring(0, pos);
		    
		    System.out.println(fileName);
		    
			ArffLoader arffLoader = new ArffLoader();
			arffLoader.setFile(files[f]);
			Instances dataset = arffLoader.getDataSet();   //读取整个数据集
			dataset.setClassIndex(dataset.numAttributes()-1);
			
			//get the minority label
			int minV = Integer.MAX_VALUE, minIndex = -1;
			int[] classCounts = dataset.attributeStats(dataset.classIndex()).nominalCounts;
		    for (int i = 0; i < classCounts.length; i++) {
		       if (classCounts[i] != 0 && classCounts[i] < minV) {
		          minV = classCounts[i];
		          minIndex = i;
		       }
		    }
		    minorityLabel = minIndex;
			
		    
		    double[] foldFmeasure = new double[nr_fold*extra_fold];
		    double[] foldAuc = new double[nr_fold*extra_fold];
		    double[] foldRecall = new double[nr_fold*extra_fold];
		    double[] foldAcc = new double[nr_fold*extra_fold];
		    double[] foldPrecision = new double[nr_fold*extra_fold];
		    double[] foldGmean = new double[nr_fold*extra_fold];
		    
		    int countI = 0;
		    for(int p=0;p<extra_fold;p++){
		    	Instances randData = new Instances(dataset, 0, dataset.numInstances());
			    randData.randomize(random);
			    randData.stratify(nr_fold);
			    
		    	for (int i=0;i<nr_fold;i++){
		    		
		    		double TP = 0;         //True positive
				    double FP = 0;         //False positive
				    double TN = 0;         //True necgtive
				    double FN = 0;         //False necgtive
				    double precision = 0;  //measure of precision
				    double recall = 0;     //measure of recall
				    ThresholdCurve tc = new ThresholdCurve();
				   
		    		Instances train = randData.trainCV(nr_fold, i);
		    		Instances test = randData.testCV(nr_fold, i);
		    		Classifier classifier = null;
		    		if (classifierType==0) {
		    			classifier = new J48();
		    		}
			    	else if (classifierType==1) {
			    		classifier = new MultilayerPerceptron();
					}
			    	else if (classifierType==2) {
						SMO smo = new SMO();
						RBFKernel kernel = new RBFKernel();
						smo.setKernel(kernel);
						classifier = smo;
					}
			    	
		    		HybridMethodV22 s = new HybridMethodV22(classifier);
		    		s.setClassifier(classifier);
		    		s.setNumIterations(m_NumberIterations);
		    		s.buildClassifier(train);
		    		minorityLabel = s.minorityLabel;
		    	
		    		FastVector mPredictions = new FastVector();				   
					for(int j=0;j<test.numInstances();j++){
						Instance instanceJ = test.instance(j);
						double r = s.classifyInstance(instanceJ);
						double[] dist = s.distributionForInstance(instanceJ);
						mPredictions.addElement(new NominalPrediction(instanceJ.classValue(), dist, instanceJ.weight()));
						
						if (instanceJ.classValue() == minorityLabel) {
							if (instanceJ.classValue() == r) {
								TP += 1;
							}
							else{
								FN += 1;
							}
						}
						else {
							if (instanceJ.classValue() == r) {
								TN += 1;
							}
							else {
								FP += 1;
							}
						}
					}
					
					Instances result = tc.getCurve(mPredictions, (int)minorityLabel);
			    	foldAuc[countI] = ThresholdCurve.getROCArea(result);
				    foldPrecision[countI] = TP/(TP+FP);
				    foldRecall[countI] = TP/(TP+FN);
				    foldFmeasure[countI] = 2*precision*recall/(precision+recall);
				    foldAcc[countI] = (TP+TN)/(TP+TN+FP+FN);
				    
				    double TPrate = TP / (TP + FN);
				    double TNrate = TN / (TN + FP);
				    foldGmean[countI] = Math.sqrt(TPrate * TNrate);
				    
				    countI += 1;
		    	}
		    }
		    
		    FileWriter writer1 = new FileWriter(path+"/"+fileName+".txt");
		    writer1.write(fileName+"\n");
		    
		    //Acc
		    double average = 0;
		    for(int i=0;i<nr_fold*extra_fold;i++){
		    	average += foldAcc[i];
		    	writer1.write("" + foldAcc[i] + "\n");
		    	//System.out.println("foldAcc="+foldAcc[i]);
		    }
		    average /= (nr_fold*extra_fold);
		    writer1.write("overallAcc=" + average + "\n");
		    rAcc[f] = average;
		    
		    //Auc
		    average = 0;
		    for(int i=0;i<nr_fold*extra_fold;i++){
		    	average += foldAuc[i];
		    	writer1.write("" + foldAuc[i] + "\n");
		    }
		    average /= (nr_fold*extra_fold);
		    writer1.write("overallAuc=" + average + "\n");
		    rAuc[f] = average;
		    rAucStd[f] = 0;
		    for(int i=0;i<nr_fold*extra_fold;i++){
		    	rAucStd[f] += Math.pow(average-foldAuc[i], 2);
		    }
		    rAucStd[f] = Math.sqrt(rAucStd[f]);
		    
		    //Fmeasure
		    average = 0;
		    for(int i=0;i<nr_fold*extra_fold;i++){
		    	average += foldFmeasure[i];
		    	writer1.write("" + foldFmeasure[i] + "\n");
		    }
		    average /= (nr_fold*extra_fold);
		    writer1.write("overallFmeasure=" + average + "\n");
		    rFmeasure[f] = average;
		    
		    //Gmean
		    average = 0;
		    for(int i=0;i<nr_fold*extra_fold;i++){
		    	average += foldGmean[i];
		    	writer1.write("" + foldGmean[i] + "\n");
		    }
		    average /= (nr_fold*extra_fold);
		    writer1.write("overallGmean=" + average + "\n");
		    rGmean[f] = average;
		    rGmeanStd[f] = 0;
		    for(int i=0;i<nr_fold*extra_fold;i++){
		    	rGmeanStd[f] += Math.pow(average-foldGmean[i], 2);
		    }
		    rGmeanStd[f] = Math.sqrt(rGmeanStd[f]);
		    
		    //Precision
		    average = 0;
		    for(int i=0;i<nr_fold*extra_fold;i++){
		    	average += foldPrecision[i];
		    	writer1.write("" + foldPrecision[i] + "\n");
		    }
		    average /= (nr_fold*extra_fold);
		    writer1.write("overallPrecision=" + average + "\n");
		    rPrecision[f] = average;
		    
		    fileNames[f] = fileName;
		    //Recall
		    average = 0;
		    for(int i=0;i<nr_fold*extra_fold;i++){
		    	average += foldRecall[i];
		    	writer1.write("" + foldRecall[i] + "\n");
		    }
		    average /= (nr_fold*extra_fold);
		    writer1.write("overallRecall=" + average + "\n");
		    rRecall[f] = average;
		   
		    writer1.flush();
		    writer1.close();
		    System.out.println(fileName+"   overall acc="+rAcc[f] + "  auc="+rAuc[f] + "  Gmean="+rGmean[f]);
		    
		    FileWriter writer2 = new FileWriter(path+"/"+"ZZfinal.txt");
			writer2.write("fileName\n");
			for(int i=0;i<=f;i++){
				writer2.write(fileNames[i]+"\n");
			}
			writer2.write("\n\nAcc\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rAcc[i]+"\n");
			}
			writer2.write("\n\nAUC\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rAuc[i]+"\n");
			}
			
			writer2.write("\n\nAUC+std\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rAuc[i]+"+" + rAucStd[i] + "\n");
			}
			
			writer2.write("\n\nFmeasure\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rFmeasure[i]+"\n");
			}
			
			writer2.write("\n\nGmean\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rGmean[i]+"\n");
			}
			
			writer2.write("\n\nGmean+std\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rGmean[i]+"+" + rGmeanStd[i] + "\n");
			}
			
			writer2.write("\n\nPrecision\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rPrecision[i]+"\n");
			}
			writer2.write("\n\nRecall\n");
			for(int i=0;i<=f;i++){
				writer2.write(""+rRecall[i]+"\n");
			}
			writer2.flush();
			writer2.close();
		}
	}
}
package bigcircle;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

//V13的基础：（1）修改margin计算方式；（2）增加权重大的样本被选中的机会
public class HybridMethodV22{
	protected Classifier m_Classifier = new ZeroR();
	protected Classifier[] m_Classifiers;
	protected int m_NumIterations = 50;
	protected int m_Seed = 1;
	protected int mKNN = 5;
	
	private static int MAX_NUM_RESAMPLING_ITERATIONS = 5;
	 
	private double[] m_Betas;
	private Random random = null;
	private double[][] classificationResult = null;
	private double[][] accumaMargin = null;
	private double[] currentMargin = null;
	private double[] noisyValues;
	private double avgNoisyValue;
	private int iterationThreshold = 5;
	
	private int[] classCounts = null;
	private int majorityNumber = 0;
	private int minorityNumber = 0;
	public double minorityLabel = 0;
	public double majorityLabel = 0;
	    
	private double selectedMajorPor = 0.3;
	private Map vdmMapofInputFeature = null;    //在input space中的特征距离
	private Map vdmMapofFeatureSpace = null;    //在feature space中的特征距离
	
	private int[] previousMarginIndex;
	
	private double marginPosThreshold = 0.5;
	
	public void buildClassifier(Instances prob) throws Exception{
		Instances data = new Instances(prob, 0, prob.numInstances());
		
		initParameter(data);
		//step 2: training
		for(int i=0;i<m_NumIterations;i++){
			//System.out.println("classifier["+i+"]");
			Instances trainData = GenerateInstances(data, random, i);
			m_Classifiers[i].buildClassifier(trainData);
			m_Betas[i] = 1;
			setWeights(data, i);
			System.out.println("classifier["+i+"]");
		}
	}
	
	
	
	private Instances GenerateInstances(Instances data, Random random, int iteration) throws Exception{
		Instances newData = new Instances(data, 0, 0);
		
		double upper = 0;
		
		double[] weights = new double[data.numInstances()];
		for(int i=0;i<weights.length;i++){
			weights[i] = 1;
		}
		int[] sampled = SelectedIndex(data, weights, random);
		int minorityCount = 0;
		int uniqueMinorityCount = 0;
		
		for(int i=0;i<weights.length;i++){
			weights[i] = Math.exp(-currentMargin[i]);
			Instance instanceI = data.instance(i);
			if (instanceI.classValue() == minorityLabel) {
        		if(sampled[i]>0){
        			minorityCount+=sampled[i];
        		}
        		else{
        			minorityCount++;
        		}
        		uniqueMinorityCount++;
			}
		}
		
		Instances majorInstance = MajoritySelected(data, sampled, weights, random);
		
		for(int i=0;i<majorInstance.numInstances();i++){
			newData.add(majorInstance.instance(i));
		}
		
		int minorityAlreadyGenerated = 0;
		for(int i=0;i<sampled.length;i++){
			Instance instanceI = data.instance(i);
			if(instanceI.classValue() == minorityLabel){
				if(sampled[i]>0){
					for(int j=0;j<sampled[i];j++){
						newData.add(instanceI);
					}
				}
				else{
					newData.add(instanceI);
				}
				minorityAlreadyGenerated++;
				int gap = majorInstance.numInstances() - minorityCount;
				
				if(gap<0){
					continue;
				}
				
				int neighbor = gap/uniqueMinorityCount;
				if(gap%uniqueMinorityCount >= minorityAlreadyGenerated){
					neighbor++;
				}
				Instances neighborInstances = GetNeighbor(data, i, neighbor, iteration, random);
				for(int j=0;j<neighborInstances.numInstances();j++){
					newData.add(neighborInstances.instance(j));
				}
			}
		}
		
		/*
		int[] numPerClass = new int[data.numClasses()];
		for(int i=0;i<numPerClass.length;i++){
			numPerClass[i] = 0;
		}
		
		for(int i=0;i<newData.numInstances();i++){
			Instance instanceI = newData.instance(i);
			numPerClass[(int)instanceI.classValue()]+=1;
		}
		upper = 1.0*(majorInstance.numInstances()+minorityCount) / data.numInstances();
		
		
		System.out.println("总共的数目="+data.numInstances()+"  选中的数量="+newData.numInstances() + "  比例="+upper + " 多类样本="+majorityLabel);
		
		for(int i=0;i<numPerClass.length;i++){
			System.out.println("最终训练集该类样本的数量="+numPerClass[i]);
		}
		*/
		return newData;
	}
	
	//根据distanceMarix中存储的距离，以data[pos]为种子，使用crossover产生新instance
	private Instances GetNeighbor(Instances data, int pos, int number, int iteration, Random random) {
		Instance instanceJ = data.instance(pos);
		List distanceToInstance = new LinkedList();
		
		for(int i=0;i<data.numInstances();i++) {
			Instance instanceI = data.instance(i);
			if(i!=pos && instanceJ.classValue() != instanceI.classValue()) {
				double d = DistanceBetweenInstances(data, pos, i, iteration);
				distanceToInstance.add(new Object[]{d, i});
			}
		}
		
		// sort the neighbors according to distance
	    Collections.sort(distanceToInstance, new Comparator() {
	        public int compare(Object o1, Object o2) {
	          double distance1 = (Double) ((Object[]) o1)[0];
	          double distance2 = (Double) ((Object[]) o2)[0];
	          return Double.compare(distance1, distance2);
	        }
	    });
		
		Instances result = new Instances(data, 0, 0); 
		int len = mKNN<distanceToInstance.size()?mKNN:distanceToInstance.size();
		for(int i=0;i<number;i++) {
			int inde = random.nextInt(len);
			int index1 = (int)(((Object[])distanceToInstance.get(inde))[1]);
			Instance instanceI = data.instance(index1);
			Instance tInstance = GenerateInstanceWithMixup(instanceJ, instanceI, random);
			result.add(tInstance);
		}
		
		return result;
	}
	
	private void setWeights(Instances data, int iteration) throws Exception{
		currentMargin = new double[data.numInstances()];
		int[] currentMarginIndex = new int[data.numInstances()];
		
		for(int i=0;i<data.numInstances();i++){
	    	Instance instance = data.instance(i);
	    	double maxValue=Double.MIN_VALUE, maxValue2=Double.MIN_VALUE,sums=0;
			int maxIndex = 0, maxIndex2 = 0;
			
			double r = m_Classifiers[iteration].classifyInstance(instance);
			
			double[] dist = m_Classifiers[iteration].distributionForInstance(instance);
			/*
			for(int j=0;j<dist.length;j++) {
				accumaMargin[i][j] += dist[j];
			}
			*/
			accumaMargin[i][(int)r] += m_Betas[iteration];
			classificationResult[i][iteration] = r;
	    	
	    	maxValue=Double.MIN_VALUE;
	    	maxValue2=Double.MIN_VALUE;
	    	sums=0;
			maxIndex = 0;
			maxIndex2 = 0;
			
			for(int j=0;j<data.numClasses();j++){
				sums+=accumaMargin[i][j];
				if(accumaMargin[i][j] > maxValue){
					maxValue = accumaMargin[i][j];
					maxIndex = j;
				}
			}
			
			for(int j=0;j<data.numClasses();j++){
				if((j!=maxIndex) && (accumaMargin[i][j] > maxValue2)){
					maxValue2 = accumaMargin[i][j];
					maxIndex2 = j;
				}
			}
			currentMargin[i] = (maxValue-maxValue2)/sums;
			//System.out.println("margin="+currentMargin[i] + "   max="+maxValue + "   max2="+maxValue2);
			//currentMargin[i] = 2*accumaMargin[i][(int)instance.classValue()]/sums-1;
	    }
		
		boolean[] isUsed = new boolean[data.numInstances()];
		for(int i=0;i<isUsed.length;i++){
			isUsed[i] = false;
		}
		
		List distanceToInstance = new LinkedList();
		
		for(int i=0;i<data.numInstances();i++){
			distanceToInstance.add(new Object[]{currentMargin[i], i});
		}
		
		// sort the neighbors according to distance
	    Collections.sort(distanceToInstance, new Comparator() {
	        public int compare(Object o1, Object o2) {
	          double distance1 = (Double) ((Object[]) o1)[0];
	          double distance2 = (Double) ((Object[]) o2)[0];
	          return Double.compare(distance1, distance2);
	        }
	    });
	    
	    
		for(int i=0;i<data.numInstances();i++){
			int index1 = (int)(((Object[])distanceToInstance.get(i))[1]);
			double s = (double)(((Object[])distanceToInstance.get(i))[0]);
			currentMarginIndex[index1] = i+1;
		}
		
	    for(int i=0;i<previousMarginIndex.length;i++){
	    	previousMarginIndex[i] = currentMarginIndex[i];
	    }
	    
	    double[][] vdm = new double[data.numClasses()][data.numClasses()];
        vdmMapofFeatureSpace.put(iteration, vdm);
        int[] featureValueCounts = new int[data.numClasses()];
        int[][] featureValueCountsByClass = new int[data.classAttribute().numValues()][data.numClasses()];
        
		for(int i=0;i<data.numInstances();i++){
			int value = (int)classificationResult[i][iteration];
			int classValue = (int)data.instance(i).classValue();
			featureValueCounts[value]++;
			featureValueCountsByClass[classValue][value]++;
		}
		
		for (int valueIndex1 = 0; valueIndex1 < data.numClasses(); valueIndex1++) {
			for (int valueIndex2 = 0; valueIndex2 < data.numClasses(); valueIndex2++) {
				double sum = 0;
	            for (int classValueIndex = 0; classValueIndex < data.numClasses(); classValueIndex++) {
	                double c1i = featureValueCountsByClass[classValueIndex][valueIndex1];
	                double c2i = featureValueCountsByClass[classValueIndex][valueIndex2];
	                double c1 = featureValueCounts[valueIndex1];
	                double c2 = featureValueCounts[valueIndex2];
	                double term1 = c1i / c1;
	                double term2 = c2i / c2;
	                sum += Math.abs(term1 - term2);
	            }
	            vdm[valueIndex1][valueIndex2] = sum;
	        }
	    }
	}
	
	private Instance GenerateInstanceWithMixup(Instance instanceX, Instance instanceY, Random random) {
		double w1 = 0.;
		
		double[] values = new double[instanceX.numAttributes()];
		
		int pos = random.nextInt(instanceX.numAttributes());
		Enumeration attrEnum = instanceX.enumerateAttributes();
		//double gap = random.nextDouble()*(1-w1)+w1;
		
		while(attrEnum.hasMoreElements()) {
			Attribute attr = (Attribute) attrEnum.nextElement();
			if (!attr.equals(instanceX.classAttribute())) {
				if (attr.isNumeric()) {
					//double gap = random.nextDouble()*(1-w1)+w1;
					double gap = random.nextGaussian();
				
					double gap1 = instanceY.value(attr) - instanceX.value(attr);
					//double dif = instanceX.value(attr)*gap + instanceY.value(attr)*(1-gap);
					
					double dif = instanceX.value(attr) + gap1*gap;
					
					values[attr.index()] = dif;
	            } 
				else if (attr.isDate()) {
					double gap = random.nextDouble()*(1-w1)+w1;
					double dif = instanceX.value(attr)*gap + instanceY.value(attr)*(1-gap);
					values[attr.index()] = (long)dif;
	            }
			}
		}
		values[instanceX.classIndex()] = instanceX.classValue();
		Instance r = new Instance(1.0, values);
		
		return r;
	}
	
	private double DistanceBetweenInstances(Instances data, int x, int y, int iteration) {
		if (x==y) {
			return 0;
		}
		double distance = 0;
		double distance1 = 0;
		Instance instanceI = data.instance(x);
		Instance instanceJ = data.instance(y);
		
		Enumeration	attrEnum = data.enumerateAttributes();
	    while (attrEnum.hasMoreElements()) {
		   Attribute attr = (Attribute) attrEnum.nextElement();
		   if (!attr.equals(data.classAttribute())) {
			  double iVal = instanceI.value(attr);
			  double jVal = instanceJ.value(attr);
			  if (attr.isNumeric()) {
				  distance += Math.pow(iVal - jVal, 2);
			  } 
			  else {
				  distance += ((double[][]) vdmMapofInputFeature.get(attr))[(int) iVal][(int) jVal];
			  }
		  }
	    }
		distance = Math.pow(distance, .5);
		distance = 1.0/(1+Math.exp(-distance));
		
		if (iteration < iterationThreshold) {
			return distance;
		}
		else{
			for(int j=0;j<iteration;j++){
				//sumW += m_Betas[j]*m_Betas[j];
				//distance += m_Betas[j]* ((double[][]) vdmMapofFeatureSpace.get(j))[(int) classificationResult[x][j]][(int) classificationResult[y][j]];
				distance1 += ((double[][]) vdmMapofFeatureSpace.get(j))[(int) classificationResult[x][j]][(int) classificationResult[y][j]];
				//distance1 += (classificationResult[x][j]==(int) classificationResult[y][j]?1:0);
			}
			
			distance1 = Math.sqrt(distance1);
			distance1 = 1.0/(1+Math.exp(-distance1));
  		}
		double thisW = 0.9;
		return distance*thisW+(1-distance1)*(1-thisW);
	}
	
	private void DistanceCalculateInInputSpace(Instances data){
		Enumeration instanceEnum = data.enumerateInstances();
	    // compute Value Distance Metric matrices for nominal features
		vdmMapofInputFeature = new HashMap();
	    Enumeration attrEnum = data.enumerateAttributes();
	    while (attrEnum.hasMoreElements()) {
	      Attribute attr = (Attribute) attrEnum.nextElement();
	      if (!attr.equals(data.classAttribute())) {
	        if (attr.isNominal() || attr.isString()) {
	          double[][] vdm = new double[attr.numValues()][attr.numValues()];
	          vdmMapofInputFeature.put(attr, vdm);
	          int[] featureValueCounts = new int[attr.numValues()];
	          int[][] featureValueCountsByClass = new int[data.classAttribute().numValues()][attr.numValues()];
	          instanceEnum = data.enumerateInstances();
	          while (instanceEnum.hasMoreElements()) {
	            Instance instance = (Instance) instanceEnum.nextElement();
	            int value = (int) instance.value(attr);
	            int classValue = (int) instance.classValue();
	            featureValueCounts[value]++;
	            featureValueCountsByClass[classValue][value]++;
	          }
	          for (int valueIndex1 = 0; valueIndex1 < attr.numValues(); valueIndex1++) {
	            for (int valueIndex2 = 0; valueIndex2 < attr.numValues(); valueIndex2++) {
	              double sum = 0;
	              for (int classValueIndex = 0; classValueIndex < data.numClasses(); classValueIndex++) {
	                double c1i = featureValueCountsByClass[classValueIndex][valueIndex1];
	                double c2i = featureValueCountsByClass[classValueIndex][valueIndex2];
	                double c1 = featureValueCounts[valueIndex1];
	                double c2 = featureValueCounts[valueIndex2];
	                double term1 = c1i / c1;
	                double term2 = c2i / c2;
	                sum += Math.abs(term1 - term2);
	              }
	              vdm[valueIndex1][valueIndex2] = sum;
	            }
	          }
	        }
	      }
	    }
	}
	
	private Instances MajoritySelected(Instances data, int[] sampled, double[] weights, Random random){
		int majorCount = 0;
		int[] majorIndex = new int[data.numInstances()];
		double sumWeight = 0;
		
		List sortedWeightOfMajor = new LinkedList();
		
		for(int i=0;i<sampled.length;i++){
			Instance instanceI = data.instance(i);
			if (sampled[i] > 0 && instanceI.classValue() == majorityLabel) {
				majorIndex[majorCount++] = i;
				sumWeight += weights[i];
				sortedWeightOfMajor.add(new Object[]{weights[i], i});
			}
		}
		
		// sort the neighbors according to distance
	    Collections.sort(sortedWeightOfMajor, new Comparator() {
	        public int compare(Object o1, Object o2) {
	          double distance1 = (Double) ((Object[]) o1)[0];
	          double distance2 = (Double) ((Object[]) o2)[0];
	          return Double.compare(distance2, distance1);
	        }
	    });
		
	    Instances re = new Instances(data, 0, 0);
	    for(int i=0;i<sortedWeightOfMajor.size()*this.selectedMajorPor;i++){
	    	double rnd = random.nextDouble();
	    	double thisSumW = 0;
	    	boolean isFind = false;
	    	for(int j=0;j<sortedWeightOfMajor.size();j++){
	    		double w = (double) ((Object[]) sortedWeightOfMajor.get(j))[0];
	    		int index = (int) ((Object[]) sortedWeightOfMajor.get(j))[1];
	    		thisSumW += w;
	    		if (thisSumW/sumWeight>=rnd) {
					for(int k=0;k<sampled[index];k++){
						re.add(data.instance(index));
					}
					isFind = true;
					break;
				}
	    	}
	    	if (!isFind) {
	    		int index = (int) ((Object[]) sortedWeightOfMajor.get(sortedWeightOfMajor.size()-1))[1];
	    		re.add(data.instance(index));
			}
	    }
	    
	    return re;
	}
	
	private int[] SelectedIndex(Instances data, double[] weights, Random random){
		int[] sampled = new int[data.numInstances()];
		for(int i=0;i<sampled.length;i++){
			sampled[i] = 0;
		}
		if (weights.length != data.numInstances()) {
			throw new IllegalArgumentException("weights.length != numInstances.");
		}

	    // Walker's method, see pp. 232 of "Stochastic Simulation" by B.D. Ripley
	    double[] P = new double[weights.length];
	    System.arraycopy(weights, 0, P, 0, weights.length);
	    Utils.normalize(P);
	    double[] Q = new double[weights.length];
	    int[] A = new int[weights.length];
	    int[] W = new int[weights.length];
	    int M = weights.length;
	    int NN = -1;
	    int NP = M;
	    for (int I = 0; I < M; I++) {
           if (P[I] < 0) {
	         throw new IllegalArgumentException("Weights have to be positive.");
	       }
	       Q[I] = M * P[I];
	       if (Q[I] < 1.0) {
	         W[++NN] = I;
	       }
	       else {
	         W[--NP] = I;
	       }
	    }
	    if (NN > -1 && NP < M) {
	       for (int S = 0; S < M - 1; S++) {
	          int I = W[S];
	          int J = W[NP];
	          A[I] = J;
	          Q[J] += Q[I] - 1.0;
	          if (Q[J] < 1.0) {
	            NP++;
	          }
	          if (NP >= M) {
	            break;
	          }
	       }
	      // A[W[M]] = W[M];
	    }

	    for (int I = 0; I < M; I++) {
	      Q[I] += I;
	    }
	    
	    for (int i = 0; i < data.numInstances(); i++) {
			sampled[i] = 0;
	    }
		for (int i = 0; i < data.numInstances(); i++) {
			int ALRV;
		    double U = M * random.nextDouble();
		    int I = (int) U;
		    if (U < Q[I]) {
		        ALRV = I;
		    }
		    else {
		        ALRV = A[I];
		    }
		    //newData.add(data.instance(ALRV));
		    if (sampled != null) {
		        sampled[ALRV]++;
		    }
		    //newData.instance(newData.numInstances() - 1).setWeight(1);
		}
		
		return sampled;
	}
	
	public double classifyInstance(Instance instance) throws Exception {
	    double[] dist = distributionForInstance(instance);
	    if (dist == null) {
	        throw new Exception("Null distribution predicted");
	    }
	    switch (instance.classAttribute().type()) {
	    case Attribute.NOMINAL:
	       double max = 0;
	       int maxIndex = 0;

	       for (int i = 0; i < dist.length; i++) {
	          if (dist[i] > max) {
	             maxIndex = i;
	             max = dist[i];
	          }
	       }
	       if (max > 0) {
	           return maxIndex;
	       }
	       else {
	    	   return Instance.missingValue();
	       }
	    case Attribute.NUMERIC:
	    case Attribute.DATE:
	    	return dist[0];
	    default:
	    	return Instance.missingValue();
	    }
	}
	
	
	public double[] distributionForInstance(Instance instance) throws Exception{
		double[] sums = new double[instance.numClasses()];
		double[] newProbs;
		for (int i = 0; i < m_NumIterations; i++) {
			if (instance.classAttribute().isNumeric() == true) {
				sums[0] += m_Classifiers[i].classifyInstance(instance);
			}
			else{
				newProbs = m_Classifiers[i].distributionForInstance(instance);
				for(int j=0;j<newProbs.length;j++){
					sums[j] += newProbs[j];
				}
			}
		}
		
		if (instance.classAttribute().isNumeric() == true) {
			sums[0] /= (double)m_NumIterations;
			return sums;
		}
		else if (Utils.eq(Utils.sum(sums), 0)) {
			return sums;
		}
		else{
			Utils.normalize(sums);
			return sums;
		}
	}
	
	
	
	//used to initnize the parameters
	private void initParameter(Instances data) throws Exception{
		random = new Random(m_Seed);
		
		m_Classifiers = Classifier.makeCopies(m_Classifier, m_NumIterations);
		
		m_Betas = new double [m_NumIterations];
	    
		currentMargin = new double[data.numInstances()];
		for(int i=0;i<currentMargin.length;i++){
			currentMargin[i] = 1.0;
		}
		
		classificationResult = new double[data.numInstances()][m_NumIterations];
		vdmMapofFeatureSpace = new HashMap();
		
		for(int i=0;i<data.numInstances();i++){
			for(int j=0;j<m_NumIterations;j++){
				classificationResult[i][j] = 0;
			}
		}
		
		accumaMargin = new double[data.numInstances()][data.numClasses()];
		for(int i=0;i<data.numInstances();i++){
			for(int j=0;j<data.numClasses();j++){
				accumaMargin[i][j] = 0;
			}
		}
		
		previousMarginIndex = new int[data.numInstances()];
		for(int i=0;i<previousMarginIndex.length;i++){
			previousMarginIndex[i] = i+1;
		}
		
		noisyValues = new double[data.numInstances()];
		avgNoisyValue = 0;
		
		MyIBk myIBk = new MyIBk();
		myIBk.setKNN(mKNN);
		myIBk.buildClassifier(data);
		this.marginPosThreshold = 0;
		
		for(int i=0;i<data.numInstances();i++){
			Instance instanceI = data.instance(i);
			Instances neighbors = myIBk.getNearestNeighbors(instanceI);
			double count = 0;
			for(int j=0;j<neighbors.numInstances();j++){
				Instance instanceJ = neighbors.instance(j);
				if(instanceI.classValue() != instanceJ.classValue()){
					count += 1;
				}
			}
			noisyValues[i] = count/neighbors.numInstances();
			avgNoisyValue += noisyValues[i];
			
			if (count>0 && count<mKNN) {
				this.marginPosThreshold += 1.0;
			}
		}
		avgNoisyValue /= data.numInstances();
		this.marginPosThreshold /= data.numInstances();
		
		//System.out.println("marginThreshold="+marginPosThreshold);
		//this.marginPosThreshold = 0.5;
		
		int minV = Integer.MAX_VALUE, minIndex = -1;
		int maxV = Integer.MIN_VALUE, maxIndex = -1;
		majorityNumber = 0;
		classCounts = data.attributeStats(data.classIndex()).nominalCounts;
	    for (int i = 0; i < classCounts.length; i++) {
	       if (classCounts[i] != 0 && classCounts[i] < minV) {
	          minV = classCounts[i];
	          minIndex = i;
	       }
	       
	       if (classCounts[i] != 0 && classCounts[i] > maxV) {
	          maxV = classCounts[i];
	          maxIndex = i;
	       }
	    }
	    minorityLabel = minIndex; minorityNumber = minV;
	    majorityLabel = maxIndex; majorityNumber = maxV;
	    
	    DistanceCalculateInInputSpace(data);
		
		//System.out.println("MarginThreshold="+marginPosThreshold);
	}
	
	private double BetaRandom(double alpha, double beta, Random random){
		double u, v;
		double x, y;
		do{
			u = random.nextGaussian();
			v = random.nextGaussian();
			x = Math.pow(u, 1./alpha);
			y = Math.pow(v, 1.0/beta);
		}while(x+y>1);
		return x/(x+y);
	}
	
	public HybridMethodV22(Classifier classifier){
		m_Classifier = classifier;
	}
	
	public void setNumIterations(int iteration){
		m_NumIterations = iteration;
	}
	
	public void setClassifier(Classifier classifier){
		m_Classifier = classifier;
	}
}
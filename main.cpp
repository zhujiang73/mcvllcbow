#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <functional>
#include <queue>
#include <map>
#include <cstring>

#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>

//#define MAX_PATH 512

#define MIN_KPS    3
#define VOCA_COLS  1000

using namespace cv;
using namespace std;
// some utility functions
void MakeDir( const string& filepath );
void help( const char* progName );
void GetDirList( const string& directory, vector<string>* dirlist );
void GetFileList( const string& directory, vector<string>* filelist );

const string kVocabularyFile( "vocabulary.xml.gz" );
const string kBowImageDescriptorsDir( "bagOfWords" );
const string kSvmsDirs( "svms" );
string resultDir;

class Params {
public:
			Params(): wordCount( VOCA_COLS ), detectorType( "SURF" ),
					  descriptorType( "SURF" ), matcherType( "FlannBased" ){ }
	int		wordCount;
	string	detectorType;
	string	descriptorType;
	string	matcherType;
};

/*
 * loop through every directory
 * compute each image's keypoints and descriptors
 * train a vocabulary
 */
Mat BuildVocabulary( const string& databaseDir,
					 const vector<string>& categories,
					 const Ptr<FeatureDetector>& detector,
					 const Ptr<DescriptorExtractor>& extractor,
					 int wordCount) {
	Mat allDescriptors;
	for ( uint32_t index = 0; index != categories.size(); ++index ) {
		cout << "processing category " << categories[index] << endl;
		string currentCategory = databaseDir + '/' + categories[index];
		vector<string> filelist;
		GetFileList( currentCategory, &filelist);
		for ( vector<string>::iterator fileindex = filelist.begin(); fileindex != filelist.end(); ++fileindex ) {
			string filepath = currentCategory + '/' + *fileindex;
			Mat image = imread( filepath );
			if ( image.empty() ) {
				continue; // maybe not an image file
			}
			vector<KeyPoint> keyPoints;
			vector<KeyPoint> keyPoints01;
			Mat descriptors;
			detector -> detect( image, keyPoints01);

                        for(uint32_t i=0; i<keyPoints01.size(); i++)
                        {
                                KeyPoint  myPoint;

                                myPoint = keyPoints01[i];

                                if (myPoint.size >= MIN_KPS) keyPoints.push_back(myPoint);
                        }


			extractor -> compute( image, keyPoints, descriptors );
			if ( allDescriptors.empty() ) {
				allDescriptors.create( 0, descriptors.cols, descriptors.type() );
			}
			allDescriptors.push_back( descriptors );
		}
		cout << "done processing category " << categories[index] << endl;
	}
	assert( !allDescriptors.empty() );
	cout << "build vocabulary..." << endl;
	BOWKMeansTrainer bowTrainer( wordCount );
	Mat vocabulary = bowTrainer.cluster( allDescriptors );
	cout << "done build vocabulary..." << endl;
	return vocabulary;
}

void  opencv_llc_bow_Descriptor(Mat &image, Mat &vocabulary,  vector<KeyPoint> &key_points, Mat &llc_descriptor)
{
        //std::cout << "opencv_llc_bow_Descriptor" << std::endl;

 	Mat descriptors;

        Params params;

        Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create( params.descriptorType );

        extractor -> compute( image, key_points, descriptors );

        int     knn = 5;
        float  fbeta = 1e-4;

        vector<vector<DMatch> > matches;

        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce" );
        //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "FlannBased" );

        matcher -> knnMatch( descriptors, vocabulary, matches, knn );

        Mat  des_mat_r01;
        for (int icx=0; icx<descriptors.rows; icx++)
        {
                des_mat_r01 = descriptors.row(icx);

                vector<DMatch> &matchesv1 = matches[icx];

                Mat  mat_cbknn;

                mat_cbknn.release();


                for (int i=0; i<knn; i++)
                {
                        Mat  mat_idx01 = vocabulary.row(matchesv1[i].trainIdx);

                        mat_cbknn.push_back(mat_idx01);
                }
                //std::cout << "mat_cbknn size : " << mat_cbknn.rows << "  " << mat_cbknn.cols << std::endl;

                Mat  ll_mat = Mat::eye(knn, knn, CV_32FC1);
                Mat  z_mat = mat_cbknn - repeat(des_mat_r01, 5, 1);
                Mat  one_mat = Mat::ones(knn, 1, CV_32FC1);
                Mat  c_mat = z_mat*z_mat.t();

                float  ftrace = trace(c_mat).val[0];

                c_mat = c_mat + ll_mat*fbeta*ftrace;

                Mat  w_mat = c_mat.inv()*one_mat;

                w_mat = w_mat/sum(w_mat).val[0];

                w_mat = w_mat.t();

                for (int i=0; i<knn; i++)
                {
                        llc_descriptor.at<float>(0, matchesv1[i].trainIdx) += w_mat.at<float>(0,i);
                }
        }

        llc_descriptor = llc_descriptor/(descriptors.rows*1.0);
}


// bag of words of an image as its descriptor, not keypoint descriptors
void ComputeBowImageDescriptors( const string& databaseDir,
								 const vector<string>& categories,
								 const Ptr<FeatureDetector>& detector,
								 const Ptr<DescriptorExtractor>& extractor,
								 Ptr<BOWImgDescriptorExtractor>& bowExtractor,
								 const string& imageDescriptorsDir,
								 map<string, Mat>* samples) {
        Mat vocabulary;
        std::string vocabularyFile = resultDir + kVocabularyFile;

        FileStorage fs( vocabularyFile, FileStorage::READ );
        if ( fs.isOpened() )  fs["vocabulary"] >> vocabulary;

        std::cout << "vocabulary rows cols = " << vocabulary.rows << "  "
                        << vocabulary.cols << std::endl;

	for (uint32_t  i = 0; i != categories.size(); ++i ) {
		string currentCategory = databaseDir + '/' + categories[i];
		vector<string> filelist;
		GetFileList( currentCategory, &filelist);
		for ( vector<string>::iterator fileitr = filelist.begin(); fileitr != filelist.end(); ++fileitr ) {
			string descriptorFileName = imageDescriptorsDir + "/" + categories[i] + "_" + ( *fileitr ) + ".xml.gz";

			std::cout << "bow: " << descriptorFileName << std::endl;

			FileStorage fs( descriptorFileName, FileStorage::READ );
			Mat imageDescriptor;
			if ( fs.isOpened() ) { // already cached
				fs["imageDescriptor"] >> imageDescriptor;
			} else {
				string filepath = currentCategory + '/' + *fileitr;
				Mat image = imread( filepath );
				if ( image.empty() ) {
					continue; // maybe not an image file
				}
				vector<KeyPoint> keyPoints;
				vector<KeyPoint> keyPoints01;

				detector -> detect( image, keyPoints01 );

                                for(uint32_t i=0; i<keyPoints01.size(); i++)
                                {
                                        KeyPoint  myPoint;

                                        myPoint = keyPoints01[i];

                                        if (myPoint.size >= MIN_KPS) keyPoints.push_back(myPoint);
                                }


                                imageDescriptor = Mat::zeros(1, VOCA_COLS, CV_32F);

				opencv_llc_bow_Descriptor( image, vocabulary, keyPoints, imageDescriptor );

				//std::cout << "imageDescriptor rows cols = " << imageDescriptor.rows << "  "
                                                //<< imageDescriptor.cols << std::endl;

				fs.open( descriptorFileName, FileStorage::WRITE );
				if ( fs.isOpened() ) {
					fs << "imageDescriptor" << imageDescriptor;
				}
			}
			if ( samples -> count( categories[i] ) == 0 ) {
				( *samples )[categories[i]].create( 0, imageDescriptor.cols, imageDescriptor.type() );
			}
			( *samples )[categories[i]].push_back( imageDescriptor );
		}
	}
}

void TrainSvm( const map<string, Mat>& samples, const string& category, const CvSVMParams& svmParams, CvSVM* svm ) {
	Mat allSamples( 0, samples.at( category ).cols, samples.at( category ).type() );
	Mat responses( 0, 1, CV_32SC1 );
	//assert( responses.type() == CV_32SC1 );
	allSamples.push_back( samples.at( category ) );
	Mat posResponses( samples.at( category ).rows, 1, CV_32SC1, Scalar::all(1) );
	responses.push_back( posResponses );

	for (  map<string, Mat>::const_iterator itr = samples.begin(); itr != samples.end(); ++itr ) {
		if ( itr -> first == category ) {
			continue;
		}
		allSamples.push_back( itr -> second );
		Mat response( itr -> second.rows, 1, CV_32SC1, Scalar::all( -1 ) );
		responses.push_back( response );

	}
	svm -> train( allSamples, responses, Mat(), Mat(), svmParams );
}



void test( char* argv[] )
{
	Params params;

	string  train_dir  = argv[2];
	string  test_dir  = argv[3];
	string  sample_name = argv[4];
	string  result_dir = argv[5];

	string  imgs_dir =  test_dir + "/" + sample_name;

	string  svms_dir = result_dir + "/" + kSvmsDirs;

	Ptr<FeatureDetector> detector = FeatureDetector::create( params.detectorType );
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create( params.descriptorType );

	vector<string>  imgs_fn;
	vector<string>  names_img_class;

	GetDirList( train_dir, &names_img_class );

	GetFileList( imgs_dir, &imgs_fn );

	map<string, CvSVM*>  svms_map;

	for ( vector<string>::iterator itr = names_img_class.begin(); itr != names_img_class.end();  itr++)
	{
                string   name_ic = *itr;

		CvSVM  *psvm = new CvSVM;

		string svmFileName = svms_dir + "/" + name_ic + ".xml.gz";
		FileStorage fs( svmFileName, FileStorage::READ );
		if ( fs.isOpened() )
		{
			fs.release();
			psvm->load( svmFileName.c_str() );

			svms_map.insert(pair<string, CvSVM*>(name_ic, psvm));
                }
                else
                {
                        std::cout << "svm : " << svmFileName << " can not load " << std::endl;
                        exit(-1);
                }

                std::cout << svmFileName << std::endl;

	}

 	Mat vocabulary;
	std::string vocabularyFile =  result_dir + "/" + kVocabularyFile;

        FileStorage fs( vocabularyFile, FileStorage::READ );
	if ( fs.isOpened() )  fs["vocabulary"] >> vocabulary;

        int  num_correct = 0;

        for ( vector<string>::iterator itr = imgs_fn.begin(); itr != imgs_fn.end();  itr++)
        {
                string  category;

                string  img_fn = *itr;

                string  queryImage = imgs_dir + "/" + img_fn;

                Mat image = imread( queryImage );

                vector<KeyPoint> keyPoints;
                vector<KeyPoint> keyPoints01;

                detector -> detect( image, keyPoints01 );

                for(uint32_t i=0; i<keyPoints01.size(); i++)
                {
                        KeyPoint  myPoint;

                        myPoint = keyPoints01[i];

                        if (myPoint.size >= MIN_KPS) keyPoints.push_back(myPoint);
                }

                Mat queryDescriptor;

                queryDescriptor = Mat::zeros(1, VOCA_COLS, CV_32F);

                opencv_llc_bow_Descriptor( image, vocabulary, keyPoints, queryDescriptor );

                int sign = 0; //sign of the positive class
                float confidence = -FLT_MAX;
                for (map<string, CvSVM*>::const_iterator itr = svms_map.begin(); itr != svms_map.end(); ++itr )
                {
                        CvSVM  *psvm = itr->second;

                        if ( sign == 0 ) {
                                float scoreValue = psvm->predict( queryDescriptor, true );
                                float classValue = psvm->predict( queryDescriptor, false );
                                sign = ( scoreValue < 0.0f ) == ( classValue < 0.0f )? 1 : -1;
                        }
                        float curConfidence = sign * psvm->predict( queryDescriptor, true );
                        if ( curConfidence > confidence ) {
                                confidence = curConfidence;
                                category = itr -> first;
                        }
                }

                std::cout << queryImage << " : " << category << std::endl;

                if (sample_name == category) num_correct++;
        }

        std::cout << num_correct << " " << imgs_fn.size() << " " << num_correct*1.0/imgs_fn.size() << std::endl;
}

void  train(char* argv[])
{
	Params params;

	string databaseDir = argv[2];
	resultDir = argv[3];

	string bowImageDescriptorsDir = resultDir + kBowImageDescriptorsDir;
	string svmsDir = resultDir + kSvmsDirs;
	MakeDir( resultDir );
	MakeDir( bowImageDescriptorsDir );
	MakeDir( svmsDir );

	// key: image category name
	// value: histogram of image
	vector<string> categories;
	GetDirList( databaseDir, &categories );

	Ptr<FeatureDetector> detector = FeatureDetector::create( params.detectorType );
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create( params.descriptorType );
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( params.matcherType );

	if ( detector.empty() || extractor.empty() || matcher.empty() ) {
		cout << "feature detector or descriptor extractor or descriptor matcher cannot be created.\n Maybe try other types?" << endl;
	}

	Mat vocabulary;
	string vocabularyFile = resultDir + '/' + kVocabularyFile;
	FileStorage fs( vocabularyFile, FileStorage::READ );
	if ( fs.isOpened() ) {
		fs["vocabulary"] >> vocabulary;
	} else {
		vocabulary = BuildVocabulary( databaseDir, categories, detector, extractor, params.wordCount );
		FileStorage fs( vocabularyFile, FileStorage::WRITE );
		if ( fs.isOpened() ) {
			fs << "vocabulary" << vocabulary;
		}
	}
	Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor( extractor, matcher );
	bowExtractor -> setVocabulary( vocabulary );
	map<string, Mat> samples;//key: category name, value: histogram

	ComputeBowImageDescriptors( databaseDir, categories, detector, extractor, bowExtractor, bowImageDescriptorsDir,  &samples );

	SVMParams svmParams;
	svmParams.svm_type = CvSVM::C_SVC;
        svmParams.kernel_type = CvSVM::LINEAR;
        //svmParams.kernel_type = CvSVM::RBF;
        svmParams.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1e+3, 1e-6);

	int sign = 0; //sign of the positive class
	float confidence = -FLT_MAX;
	for ( map<string, Mat>::const_iterator itr = samples.begin(); itr != samples.end(); ++itr ) {
		CvSVM svm;
		string svmFileName = svmsDir + "/" + itr -> first + ".xml.gz";

		std::cout << "TrainSvm " <<  svmFileName  << std::endl;

                TrainSvm( samples, itr->first, svmParams, &svm );
                if (svmsDir != "") svm.save( svmFileName.c_str() );
 	}

        std::cout << "train  done" << std::endl;
}

int main( int argc, char* argv[] )
{
	cv::initModule_nonfree();
	// read params

	if ( argc < 2)
	{
		help( argv[0] );
		return -1;
	}


        string  str_cmd = argv[1];

        if (str_cmd == "train")
        {
                train(argv);
                return 0;
        }

        if (str_cmd == "test")
        {
                test(argv);
                return 0;
        }

	return 0;
}

void help( const char* progName ) {
	std::cout << "OpenCV LLC BOW TEST" << std::endl;
}

void MakeDir( const string& filepath )
{
	char path[MAX_PATH];

	strncpy(path, filepath.c_str(),  MAX_PATH);

        #ifdef _WIN32
                mkdir(path);
        #else
                mkdir(path, 0755);
        #endif
	//CreateDirectory( path, 0 );

	//int status;


        //status = mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

void ListDir( const string& directory, vector<string>* entries)
{
        //WIN32_FIND_DATA  entry;
	char dir[MAX_PATH];
	//HANDLE hFind;
	string  str_dir = directory;

	strncpy(dir, str_dir.c_str(), MAX_PATH);

        DIR             *p_dir;
        struct dirent *p_dirent;

        p_dir = opendir(dir);

        while(p_dirent = readdir(p_dir))    //读取目录输出目录项的节点号和文件名
        {
                string  str_fn = p_dirent->d_name;

                if (str_fn != "." && str_fn != "..")  entries->push_back(str_fn);
        }
}

void GetDirList( const string& directory, vector<string>* dirlist )
{
	ListDir( directory, dirlist);
}

void GetFileList( const string& directory, vector<string>* filelist )
{
	ListDir( directory, filelist);

}


//#include "stdafx.h"
#include "marker.h"
#include <thread>


using namespace cv;
using namespace std;

vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
vector<int> contourMarker;
vector<vector<Point> > contours0;
Point2f markerPts2D[4];
vector<Point3f> markerPts3D(4);
Inv_Moments srsMoments;
cv::Mat_<float> distCoeffs, cameraMatrix;
float markerSize = 71;//71 mm
Mat view, viewGray, viewGray0;
Mat frame;
int findChildNumber(vector<Vec4i> hierarchy, int childIndex) {
	int childCounter = 0;
	if (hierarchy[childIndex][2] > -1) {
		int index = hierarchy[childIndex][2];
		childCounter++;
		while (hierarchy[index][0] > -1) {
			childCounter++;
			index = hierarchy[index][0];
		}
	}
	return childCounter;
}

int findMarkerCandidate(int index = -1) {//Определяем количество детей у всех вложенных контуров
	int currentIndex;
	if (index == -1) currentIndex = 0;
	else currentIndex = hierarchy[index][2];
	int nodeCounter = 0;

	bool flagEnd = false;

	while (!flagEnd) {
		if (currentIndex > -1) {
			contourMarker[currentIndex] = findMarkerCandidate(currentIndex);
			currentIndex = hierarchy[currentIndex][0];
			nodeCounter++;
		}
		else flagEnd = true;
	}
	return nodeCounter;
}
bool findChildAndApprox(int index, float eps) { // Аппроксимация контура и его потомков
	int currentIndex;
	bool flag = true;
	approxPolyDP(Mat(contours0[index]), contours[index], eps, true);
	//	if(contours[index].size() < 4) return false; // если у контура или любого из его потомков меньше 4 точек(прямоугольник) , это не маркер
	currentIndex = hierarchy[index][2];//первый потомок
	int nodeCounter = 0;
	bool flagEnd = false;
	while (!flagEnd) {
		if (currentIndex > -1) {// если есть потомки
			flag = findChildAndApprox(currentIndex, 2.0);
			if (!flag) break;
			currentIndex = hierarchy[currentIndex][0];//следующий потомок
		}
		else flagEnd = true;
	}
	return flag;
}
int sign(double val)
{
	return (val > 0) ? (1) : ((val < 0) ? (-1) : (0));
}
void Hu_Moments(vector<Point> contour, Inv_Moments* inv_moments) {
	//double m00,m10,m01;//,m20,m11,m02,m30,m21,m12,m03;
	//double mu20,mu11,mu02,mu30,mu21,mu12,mu03;
	//double nu20,nu11,nu02,nu30,nu21,nu12,nu03;
	//	double M1,M2,M3,M4,M5,M6,M7;
	inv_moments->clear();
	int size = contour.size();
	inv_moments->m00 = size;
	//m10 = m01 = mu20 = mu11 = mu02 = mu30 = mu21 = mu12 = mu03 = 0;
	//nu20 = nu11 = nu02 = nu30 = nu21 = nu12 = nu03 = 0;

	for (int i = 0; i < size; i++) {
		inv_moments->m10 = inv_moments->m10 + contour[i].x;
		inv_moments->m01 = inv_moments->m01 + contour[i].y;
	}
	double xc, yc;
	xc = inv_moments->m10 / size;
	yc = inv_moments->m01 / size;
	for (int i = 0; i < size; i++) {
		double mx = contour[i].x - xc;
		double my = contour[i].y - yc;
		double mx2 = mx * mx;
		double my2 = my * my;
		double mxy = (contour[i].x - xc) * (contour[i].y - yc);

		inv_moments->mu20 = inv_moments->mu20 + mx2; //(contour[i].x - xc) * (contour[i].x - xc);
		inv_moments->mu02 = inv_moments->mu02 + my2; //(contour[i].y - yc) * (contour[i].y - yc);
		inv_moments->mu11 = inv_moments->mu11 + mxy; //(contour[i].x - xc) * (contour[i].y - yc);
		inv_moments->mu30 = inv_moments->mu30 + mx2 * mx;
		inv_moments->mu21 = inv_moments->mu21 + mx2 * my;
		inv_moments->mu12 = inv_moments->mu12 + mx * my2;
		inv_moments->mu03 = inv_moments->mu03 + my2 * my;
	}

	double   inv_m00 = 1. / inv_moments->m00;
	double inv_sqrt_m00 = std::sqrt(std::abs(inv_m00));
	double s2 = inv_m00 * inv_m00, s3 = s2 * inv_sqrt_m00;

	inv_moments->nu20 = inv_moments->mu20 * s2; inv_moments->nu11 = inv_moments->mu11 * s2; inv_moments->nu02 = inv_moments->mu02 * s2;
	inv_moments->nu30 = inv_moments->mu30 * s3; inv_moments->nu21 = inv_moments->mu21 * s3; inv_moments->nu12 = inv_moments->mu12 * s3; inv_moments->nu03 = inv_moments->mu03 * s3;

	double t0 = inv_moments->nu30 + inv_moments->nu12;
	double t1 = inv_moments->nu21 + inv_moments->nu03;

	double q0 = t0 * t0, q1 = t1 * t1;

	double n4 = 4 * inv_moments->nu11;
	double s = inv_moments->nu20 + inv_moments->nu02;
	double d = inv_moments->nu20 - inv_moments->nu02;

	inv_moments->hu0 = s;
	inv_moments->hu1 = d * d + n4 * inv_moments->nu11;
	inv_moments->hu3 = q0 + q1;
	inv_moments->hu5 = d * (q0 - q1) + n4 * t0 * t1;

	t0 *= q0 - 3 * q1;
	t1 *= 3 * q0 - q1;

	q0 = inv_moments->nu30 - 3 * inv_moments->nu12;
	q1 = 3 * inv_moments->nu21 - inv_moments->nu03;

	inv_moments->hu2 = q0 * q0 + q1 * q1;
	inv_moments->hu4 = q0 * t0 + q1 * t1;
	inv_moments->hu6 = q1 * t0 - q0 * t1;

	double r = sqrt(inv_moments->hu0);
	double r2 = r * r;
	double r4 = r2 * r2;
	double r6 = r4 * r2;

	inv_moments->M1 = inv_moments->hu1 / r4;
	inv_moments->M2 = inv_moments->hu2 / r6;
	inv_moments->M3 = inv_moments->hu3 / r6;
	double r12 = r6 * r6;
	inv_moments->M4 = inv_moments->hu4 / r12;
	inv_moments->M5 = inv_moments->hu5 / (r4 * r4);
	inv_moments->M6 = inv_moments->hu6 / r12;

}
void findMarker(vector<Point>& contour, int cntr_index, Mat_<float>& M, int& marker_info) {

	Point2f imgPts[4];
	vector<Vec3f> srsPts(4);
	vector<Point> MarkerCandidatePoint;
	int MCP_index = 0;

	marker_info = -1;
	imgPts[0] = contour[0];
	imgPts[1] = contour[1];
	imgPts[2] = contour[2];
	imgPts[3] = contour[3];

	Mat_<float> MPT = getPerspectiveTransform(imgPts, markerPts2D);

	for (int i = 0; i < 4; i++)
	{
		srsPts[i][0] = imgPts[i].x;
		srsPts[i][1] = imgPts[i].y;
		srsPts[i][2] = 1;
		Mat_<float> V = MPT * Mat(srsPts[i], false);
		V(0) = V(0) / V(2);
		V(1) = V(1) / V(2);
		Point pnt;
		MarkerCandidatePoint.push_back(pnt);
		MarkerCandidatePoint[MCP_index].x = V(0);
		MarkerCandidatePoint[MCP_index].y = V(1);
		MCP_index++;

	}
	int index = hierarchy[cntr_index][2];
	for (int i = 0; i < 6; i++) {
		int cntr_size = contours[index].size();
		vector<Vec3f> srsPts1(cntr_size);

		for (int j = 0; j < cntr_size; j++)
		{
			srsPts1[j][0] = contours[index][j].x;
			srsPts1[j][1] = contours[index][j].y;
			srsPts1[j][2] = 1;
			Mat_<float> V = MPT * Mat(srsPts1[j], false);
			V(0) = V(0) / V(2);
			V(1) = V(1) / V(2);
			Point pnt;
			MarkerCandidatePoint.push_back(pnt);
			MarkerCandidatePoint[MCP_index].x = V(0);
			MarkerCandidatePoint[MCP_index].y = V(1);
			MCP_index++;
		}
		index = hierarchy[index][0];
	}
	Inv_Moments invMoments;
	Hu_Moments(MarkerCandidatePoint, &invMoments);

	double sum = 0;
	double m_A = static_cast <double>(sign(srsMoments.M1)) * log10(fabs(srsMoments.M1 + 0.1e-60));
	double m_B = static_cast <double>(sign(invMoments.M1)) * log10(fabs(invMoments.M1 + 0.1e-60));
	sum = sum + fabs(m_A - m_B);
	m_A = static_cast <double>(sign(srsMoments.M2)) * log10(fabs(srsMoments.M2 + 0.1e-60));
	m_B = static_cast <double>(sign(invMoments.M2)) * log10(fabs(invMoments.M2 + 0.1e-60));
	sum = sum + fabs(m_A - m_B);
	m_A = static_cast <double>(sign(srsMoments.M3)) * log10(fabs(srsMoments.M3 + 0.1e-60));
	m_B = static_cast <double>(sign(invMoments.M3)) * log10(fabs(invMoments.M3 + 0.1e-60));
	sum = sum + fabs(m_A - m_B);
	m_A = static_cast <double>(sign(srsMoments.M4)) * log10(fabs(srsMoments.M4 + 0.1e-60));
	m_B = static_cast <double>(sign(invMoments.M4)) * log10(fabs(invMoments.M4 + 0.1e-60));
	sum = sum + fabs(m_A - m_B);
	m_A = static_cast <double>(sign(srsMoments.M5)) * log10(fabs(srsMoments.M5 + 0.1e-60));
	m_B = static_cast <double>(sign(invMoments.M5)) * log10(fabs(invMoments.M5 + 0.1e-60));
	sum = sum + fabs(m_A - m_B);
	m_A = static_cast <double>(sign(srsMoments.M6)) * log10(fabs(srsMoments.M6 + 0.1e-60));
	m_B = static_cast <double>(sign(invMoments.M6)) * log10(fabs(invMoments.M6 + 0.1e-60));
	sum = sum + fabs(m_A - m_B);

	if (sum < 2) {
		for (int a = 0; a < 4; a++) {
			if (a > 0) { // Для первой итерации трансформация не нужна
				for (int k = 0; k < MarkerCandidatePoint.size(); k++) {
					int px = MarkerCandidatePoint[k].y;  //поворачиваем на 90 градусов
					int py = -MarkerCandidatePoint[k].x;
					MarkerCandidatePoint[k].x = px;//возвращаем на место
					MarkerCandidatePoint[k].y = py;
				}
			}
			Hu_Moments(MarkerCandidatePoint, &invMoments);//пересчитываем моменты для повернутого маркера

			double sum_n = 0;
			m_A = static_cast <double>(sign(srsMoments.nu20)) * log(fabs(srsMoments.nu20 + 0.1e-60));
			m_B = static_cast <double>(sign(invMoments.nu20)) * log(fabs(invMoments.nu20 + 0.1e-60));
			sum_n = sum_n + fabs(1 / m_A - m_B);
			m_A = static_cast <double>(sign(srsMoments.nu11)) * log(fabs(srsMoments.nu11 + 0.1e-60));
			m_B = static_cast <double>(sign(invMoments.nu11)) * log(fabs(invMoments.nu11 + 0.1e-60));
			sum_n = sum_n + fabs(m_A - m_B);
			m_A = static_cast <double>(sign(srsMoments.nu02)) * log(fabs(srsMoments.nu02 + 0.1e-60));
			m_B = static_cast <double>(sign(invMoments.nu02)) * log(fabs(invMoments.nu02 + 0.1e-60));
			sum_n = sum_n + fabs(m_A - m_B);
			m_A = static_cast <double>(sign(srsMoments.nu30)) * log(fabs(srsMoments.nu30 + 0.1e-60));
			m_B = static_cast <double>(sign(invMoments.nu30)) * log(fabs(invMoments.nu30 + 0.1e-60));
			sum_n = sum_n + fabs(m_A - m_B);
			m_A = static_cast <double>(sign(srsMoments.nu21)) * log(fabs(srsMoments.nu21 + 0.1e-60));
			m_B = static_cast <double>(sign(invMoments.nu21)) * log(fabs(invMoments.nu21 + 0.1e-60));
			sum_n = sum_n + fabs(m_A - m_B);
			m_A = static_cast <double>(sign(srsMoments.nu12)) * log(fabs(srsMoments.nu12 + 0.1e-60));
			m_B = static_cast <double>(sign(invMoments.nu12)) * log(fabs(invMoments.nu12 + 0.1e-60));
			sum_n = sum_n + fabs(m_A - m_B);
			m_A = static_cast <double>(sign(srsMoments.nu03)) * log(fabs(srsMoments.nu03 + 0.1e-60));
			m_B = static_cast <double>(sign(invMoments.nu03)) * log(fabs(invMoments.nu03 + 0.1e-60));
			sum_n = sum_n + fabs(m_A - m_B);

			if (sum_n < 20) {
				vector<Point2f> cntrPts(4); //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Может быть ошибка - не известно сколько точек в контуре!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
				for (int t = 0; t < contours[cntr_index].size(); t++) {
					cntrPts[t] = contours[cntr_index][a];
					if (++a > 3) a = 0;
				}
				cv::Mat raux, taux;
				solvePnP(markerPts3D, cntrPts, cameraMatrix, distCoeffs, raux, taux);
				cv::Mat_<float> Rvec;
				cv::Mat_<float> Tvec;
				raux.convertTo(Rvec, CV_32F);
				taux.convertTo(Tvec, CV_32F);
				cv::Mat_<float> rotMat(3, 3);
				cv::Mat_<float> TransformMat = Mat::eye(3, 4, CV_32F);

				cv::Rodrigues(Rvec, rotMat);
				for (int col = 0; col < 3; col++) {
					for (int row = 0; row < 3; row++) {
						TransformMat(row, col) = rotMat(row, col);
					}
				}

				TransformMat(0, 3) = Tvec(0);
				TransformMat(1, 3) = Tvec(1);
				TransformMat(2, 3) = Tvec(2);
				M = cameraMatrix * TransformMat;

				marker_info = 0;
				for (int y0 = -10; y0 < 15; y0 += 10)
					for (int x0 = -10; x0 < 15; x0 += 10) {
						int sum_pnt = 0;
						for (int xn = -2; xn < 5; xn += 2)
							for (int yn = -2; yn < 5; yn += 2) {
								Vec4f point;
								point[0] = x0 + xn;
								point[1] = y0 + yn;
								point[2] = 0;
								point[3] = 1;
								cv::Mat_<float> V = M * Mat(point, false);
								Point pnt;
								pnt.x = V(0) / V(2);
								pnt.y = V(1) / V(2);
								sum_pnt += viewGray.at<unsigned char>(pnt);
							}
						if (sum_pnt < 510) {
							int bit_count = (-y0 / 10 + 1) * 3 + (-x0 / 10 + 1);
							int mask = 1;
							mask = mask << bit_count;
							marker_info = marker_info | mask;
						}
					}
				break;
			}
		}
	}
}
#pragma warning(disable:4996)
int main(void)
{
	int cameraId = 0;
	VideoCapture capture;
	bool flagLegalChild = false;
	vector<bool> legalChild;
	double Coeffs[5] = { -4.8049913452186989e-002, -2.5485069579953956e-001,-1.0408791297464905e-004, -6.8076470390628259e-003, 3.4773829851554732e+000 };
	double Matrix[9] = { 6.8460821965519756e+002, 0., 2.9838945057018270e+002,
		0.,6.8497133298845097e+002, 2.4162740291007879e+002,
		0., 0., 1. };

	distCoeffs = Mat(5, 1, CV_64F, Coeffs);
	cameraMatrix = Mat(3, 3, CV_64F, Matrix);

	//capture.open(cameraId);
	capture = VideoCapture(0, CAP_DSHOW);

	if (!capture.isOpened())  return fprintf(stderr, "Could not initialize video (%d) capture\n", cameraId), -2;

	if (capture.isOpened())
		printf("%s", "Camera OK.");
	const char* civ = "Color Image View";
	const char* giv = "Gray Image View";
	namedWindow(civ, WINDOW_AUTOSIZE);
	namedWindow(giv, WINDOW_AUTOSIZE);

	FILE* file;
	if ((file = fopen("data.txt", "rb+")) == NULL) {
		printf("Cannot open file.");
		return 1;
	}

	fseek(file, 0, SEEK_SET);
	fread(&srsMoments, sizeof(Inv_Moments), 1, file);
	fclose(file);

	int levels = 7;

	int threshold_value = 128;
	int threshold_type = 0;
	int const max_value = 255;
	int const max_type = 4;
	int const max_BINARY_value = 255;

	markerPts3D[0].x = -markerSize / 2;				markerPts3D[0].y = -markerSize / 2;		markerPts3D[0].z = 0;
	markerPts3D[1].x = markerSize / 2;				markerPts3D[1].y = -markerSize / 2;		markerPts3D[1].z = 0;
	markerPts3D[2].x = markerSize / 2;				markerPts3D[2].y = markerSize / 2;		markerPts3D[2].z = 0;
	markerPts3D[3].x = -markerSize / 2;				markerPts3D[3].y = markerSize / 2;		markerPts3D[3].z = 0;

	markerPts2D[0].x = -markerSize / 2;				markerPts2D[0].y = -markerSize / 2;
	markerPts2D[1].x = markerSize / 2;				markerPts2D[1].y = -markerSize / 2;
	markerPts2D[2].x = markerSize / 2;				markerPts2D[2].y = markerSize / 2;
	markerPts2D[3].x = -markerSize / 2;				markerPts2D[3].y = markerSize / 2;
	for (int ir = 0;; ir++) {

		capture >> frame;
		cvtColor(frame, viewGray0, COLOR_BGR2GRAY);

		threshold(viewGray0, viewGray, threshold_value, max_BINARY_value, threshold_type);

		//		cv::adaptiveThreshold(viewGray0,viewGray,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY_INV,7,7);

		findContours(viewGray, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		contours.resize(contours0.size());
		contourMarker.resize(contours0.size(), 0);// 0 means contour don't have child
		legalChild.resize(contours0.size(), false);
		int h_s = hierarchy.size();
		if (h_s == contours0.size() && h_s > 0) 	int test = findMarkerCandidate();

		Point2f imgPts[4];
		vector<vector<vector<Vec3f>>> OrthoMarker;
		vector<int> marker_contour;
		//		int store_k;

		vector<Mat> marker_Mat(contours0.size());

		for (size_t k = 0; k < contours0.size(); k++)
			if (contourMarker[k] == 6) {//если потомков 6
				legalChild[k] = findChildAndApprox(k, 4.0);
			}

		//			cvtColor(frame, viewGray, COLOR_BGR2GRAY);

		threshold(viewGray0, viewGray, threshold_value, max_BINARY_value, threshold_type);

		int OrtMrkr_index = -1;//При первом увеличении на 1 будет 0

		for (size_t k = 0; k < contours0.size(); k++) {
			if (contourMarker[k] == 6 && contours[k].size() == 4 && legalChild[k]) {
				int t = k;
				marker_contour.push_back(t);

			}
		}
		int cntr_size;
		cntr_size = marker_contour.size();
		if (cntr_size > 0) {
			vector<thread> thrd(cntr_size);
			vector<int> mi(cntr_size);
			vector<Mat_<float>> Matrix(cntr_size);
			for (int k = 0; k < cntr_size - 1; k++) {
				thrd[k] = thread(findMarker, ref(contours[marker_contour[k]]), marker_contour[k], ref(Matrix[k]), ref(mi[k]));
			}
			cntr_size--;
			if (cntr_size >= 0) findMarker(contours[marker_contour[cntr_size]], marker_contour[cntr_size], Matrix[cntr_size], mi[cntr_size]);
			for (int k = 0; k < cntr_size; k++) {
				thrd[k].join();
			}
			cntr_size++;
			for (int k = 0; k < cntr_size; k++) {
				if (mi[k] >= 0) {
					Vec4f point0;
					point0[0] = 0;
					point0[1] = 0;
					point0[2] = 0;
					point0[3] = 1;
					cv::Mat_<float> MT = Matrix[k];
					cv::Mat_<float> V = MT * Mat_<float>(point0, false);
					Vec4f pointx;
					pointx[0] = 30;
					pointx[1] = 0;
					pointx[2] = 0;
					pointx[3] = 1;
					cv::Mat_<float> Vx = MT * Mat(pointx, false);
					Vec4f pointy;
					pointy[0] = 0;
					pointy[1] = 30;
					pointy[2] = 0;
					pointy[3] = 1;
					cv::Mat_<float> Vy = MT * Mat(pointy, false);
					Vec4f pointz;
					pointz[0] = 0;
					pointz[1] = 0;
					pointz[2] = -30;
					pointz[3] = 1;
					cv::Mat_<float> Vz = MT * Mat(pointz, false);

					Point pnt0, pntx, pnty, pntz;
					pnt0.x = V(0) / V(2);
					pnt0.y = V(1) / V(2);
					pntx.x = Vx(0) / Vx(2);
					pntx.y = Vx(1) / Vx(2);
					pnty.x = Vy(0) / Vy(2);
					pnty.y = Vy(1) / Vy(2);
					pntz.x = Vz(0) / Vz(2);
					pntz.y = Vz(1) / Vz(2);

					arrowedLine(frame, pnt0, pntx, Scalar(0, 255, 0), 1);
					arrowedLine(frame, pnt0, pnty, Scalar(0, 0, 255), 1);
					arrowedLine(frame, pnt0, pntz, Scalar(255, 0, 0), 1);
				}
			}
		}
		imshow(giv, viewGray);

		imshow(civ, frame);

		if (waitKey(1) >= 0) break;
	}
	return 0;
}


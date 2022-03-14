#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <iterator>

#include <InEKF/Core>
#include <InEKF/Inertial>
#include <tqdm.h>
#include <matplot/matplot.h>
namespace plt = matplot;

std::vector<double> to_std_vec(Eigen::MatrixXd v){
    return std::vector<double>(v.data(), v.data()+v.size());
}

Eigen::VectorXd strToVec(std::string in){
    // turn each line into a vector
    std::istringstream is( in );
    std::vector<double> line( ( std::istream_iterator<double>( is ) ), ( std::istream_iterator<double>() ) );
    
    // turn each vector into an eigen vector
    auto n = line.size();
    Eigen::Map<Eigen::VectorXd> data(&line[0], n);

    return data;
}

int main(){
    // Set up initial state
    Eigen::Matrix3d R0;
    R0 << 0.00000, 0.99863, -0.05234,
         -0.99452, 0.00547, 0.10439,
          0.10453, 0.05205, 0.99316;
    InEKF::SO3 Rot(R0);
    Eigen::Matrix<double,12,1> xi;
    xi << 0, 0, -4.66134,       // velocity
        -0.077, -0.02, -2.2082, // position
        0,0,0,0,0,0;            // bias
    Eigen::Matrix<double,15,1> s;
    s << 0.274156, 0.274156, 0.274156, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.000025, 0.000025, 0.000025, 0.0025, 0.0025, 0.0025;
    InEKF::SE3<2,6> state(Rot, xi, s.asDiagonal());

    // Set up DVL
    Eigen::Matrix3d dvlR;
    dvlR << 0.000, -0.995, 0.105,
            0.999, 0.005, 0.052,
            -0.052, 0.104, 0.993;
    Eigen::Vector3d dvlT;
    dvlT << -0.17137, 0.00922, -0.33989;
    InEKF::DVLSensor dvl(dvlR, dvlT);
    dvl.setNoise(.0101*2.6, .005*(3.14/180)*sqrt(200.0));

    // Set up Depth sensor
    InEKF::DepthSensor depth(51.0 * (1.0/100) * (1.0/2));

    // Set up IEKF
    InEKF::InEKF<InEKF::InertialProcess> iekf(state, InEKF::RIGHT);
    iekf.addMeasureModel("DVL", &dvl);
    iekf.addMeasureModel("Depth", &depth);

    // Set up Inertial noise
    iekf.pModel->setGyroNoise( .005 *  (3.14/180)  * sqrt(200.0) );
    iekf.pModel->setAccelNoise( 20.0 * (pow(10, -6)/9.81) * sqrt(200.0) );
    iekf.pModel->setGyroBiasNoise(0.001);
    iekf.pModel->setAccelBiasNoise(0.001);


    // Iterate through all of data!
    int n = 3000; // underwater_data goes to about ~3900
    int i = 0;
    std::ifstream infile("../data/underwater_data.txt");
    if(infile.fail()) std::cout << "Failed to open file" << std::endl;
    std::string line;
    Eigen::Matrix<double,6,1> imu_data = Eigen::Matrix<double,6,1>::Zero();
    Eigen::Matrix<double,6,1> dvl_data = Eigen::Matrix<double,6,1>::Zero();
    Eigen::Matrix<double,1,1> depth_data = Eigen::Matrix<double,1,1>::Zero();
    Eigen::MatrixXd v(n,3), p(n,3), R(3*n,3);
    Eigen::MatrixXd v_result(n,3), p_result(n,3), R_result(3*n,3);

    std::string type;
    Eigen::VectorXd data;
    int split;
    double dt = 0;
    tqdm bar;
    while (getline(infile, line)){
        split = line.find(" ");
        type = line.substr(0, split);
        data = strToVec(line.substr(split+1, line.size()));

        // PREDICTION STEP
        if (type == "IMU"){
            assert((data.size()-1) == 6);
            dt = data[0];
            imu_data = data.tail(6);
            state = iekf.Predict(imu_data, dt);
        }

        // UPDATE STEP
        else if (type == "DVL"){
            assert((data.size()-2) == 3);
            dvl_data << data[1], data[2], data[3],
                        imu_data[0], imu_data[1], imu_data[2];
            state = iekf.Update(dvl_data, "DVL");            
        }
        else if (type == "DEPTH"){
            assert((data.size()-1) == 1);
            depth_data = data.tail(1);
            state = iekf.Update(depth_data, "Depth");

            // DVL is last in list of data saved, so we save our state here
            R_result.block<3,3>(3*i,0) = state.R()();
            // save v in local frame
            v_result.row(i) = (state.R()().transpose() * state[0]).transpose();
            p_result.row(i) = state[1].transpose();
            i += 1;
        }

        // GROUND TRUTH TO COMPARE TO
        else if (type == "P"){
            assert((data.size()-1) == 3);
            p.row(i) = data.tail(3);
        }
        else if (type == "V"){
            assert((data.size()-1) == 3);
            v.row(i) = data.tail(3);
            // put v into local frame
            v.row(i) = ( R.block<3,3>(3*i,0).transpose() * v.row(i).transpose() ).transpose().eval();
        }
        else if (type == "R"){
            assert((data.size()-1) == 9);
            R.block<3,3>(3*i,0) << data.segment<3>(1), data.segment<3>(4), data.segment<3>(7);
            R.block<3,3>(3*i,0).transposeInPlace();
        }

        if(i >= n){
            break;
        }

        bar.progress(i,n);
    }
    bar.finish();
    

    // Plot everything
    Eigen::VectorXd t1 = Eigen::VectorXd::LinSpaced(n, 0, n*dt);
    std::vector<double> t(t1.data(), t1.data()+n);
    plt::figure(true)->size(1000,1000);

    // Global position
    plt::subplot(3, 3, 0);
    plt::hold(true);
    plt::ylabel("Position");
    plt::title("X (global)");
    plt::plot(t, to_std_vec(p.col(0)))->display_name("Actual");
    plt::plot(t, to_std_vec(p_result.col(0)))->display_name("Result");
    plt::subplot(3, 3, 1);
    plt::hold(true);
    plt::title("Y (global)");
    plt::plot(t, to_std_vec(p.col(1)));
    plt::plot(t, to_std_vec(p_result.col(1)));
    plt::subplot(3, 3, 2);
    plt::hold(true);
    plt::title("Z (global)");
    plt::plot(t, to_std_vec(p.col(2)));
    plt::plot(t, to_std_vec(p_result.col(2)));

    // Local Velocity
    plt::subplot(3, 3, 3);
    plt::hold(true);
    plt::ylabel("Velocity");
    plt::title("X (local)");
    plt::plot(t, to_std_vec(v.col(0)));
    plt::plot(t, to_std_vec(v_result.col(0)));
    plt::subplot(3, 3, 4);
    plt::hold(true);
    plt::title("Y (local)");
    plt::plot(t, to_std_vec(v.col(1)));
    plt::plot(t, to_std_vec(v_result.col(1)));
    plt::subplot(3, 3, 5);
    plt::hold(true);
    plt::title("Z (local)");
    plt::plot(t, to_std_vec(v.col(2)));
    plt::plot(t, to_std_vec(v_result.col(2)));

    // All the angles
    plt::subplot(3,3,6);
    plt::hold(true);
    plt::ylabel("Angles");
    plt::title("Pitch");
    Eigen::VectorXd angle(n);
    Eigen::VectorXd angle_result(n);
    for(int j=0;j<n;j++){
        angle[j] = -std::asin(R(3*j+2,0));
        angle_result[j] = -std::asin(R_result(3*j+2,0));
    }
    plt::plot(t, to_std_vec(angle));
    plt::plot(t, to_std_vec(angle_result));

    plt::subplot(3,3,7);
    plt::hold(true);
    plt::title("Roll");
    for(int j=0;j<n;j++){
        angle[j] = std::atan2(R(3*j+2,1), R(3*j+2,2));
        angle_result[j] = std::atan2(R_result(3*j+2,1), R_result(3*j+2,2));
    }
    plt::plot(t, to_std_vec(angle));
    plt::plot(t, to_std_vec(angle_result));

    plt::subplot(3,3,8);
    plt::hold(true);
    plt::title("Yaw");
    for(int j=0;j<n;j++){
        angle[j] = std::atan2(R(3*j+1,0), R(3*j,0));
        angle_result[j] = std::atan2(R_result(3*j+1,0), R_result(3*j,0));
    }
    plt::plot(t, to_std_vec(angle));
    plt::plot(t, to_std_vec(angle_result));

    plt::show();

    return 0;
}
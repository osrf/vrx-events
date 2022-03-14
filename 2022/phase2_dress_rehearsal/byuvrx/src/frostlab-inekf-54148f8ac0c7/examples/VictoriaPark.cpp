#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <tuple>
#include <time.h>
#include <unistd.h>

#include <InEKF/Core>
#include <InEKF/SE2Models>
#include <tqdm.h>
#include <matplot/matplot.h>
namespace plt = matplot;

typedef InEKF::SE2<Eigen::Dynamic> SE2_D;

typedef std::pair<int, double> sort_pair;
Eigen::VectorXi solveCostMatrix(Eigen::MatrixXd M){
    int n_mm = M.rows();
    Eigen::VectorXi result(n_mm);

    for(int i=0;i<n_mm;i++){
        Eigen::MatrixXd::Index minMM, minLM;
        M.minCoeff(&minMM, &minLM);
        result[minMM] = minLM;
        M.col(minLM).setConstant(1e8);
        M.row(minMM).setConstant(1e8);
    }
    return result;
}

Eigen::VectorXi dataAssocation(SE2_D state, Eigen::VectorXd zs, InEKF::LandmarkSensor ls){
    int n_lm = state().cols() - 2 - 1;
    int n_mm = zs.size() / 2;

    if(n_lm == 0){
        return Eigen::VectorXi::Ones(n_mm)*-1;
    }

    // hardcode in chi2 values
    double alpha = 5.991464547107979;
    double beta  = 9.21034037197618;
    Eigen::MatrixXd M = Eigen::MatrixXd::Constant(n_mm, n_lm+n_mm, alpha);
    // Make cost matrix
    for(int i=0;i<n_lm;i++){
        ls.sawLandmark(i, state);
        for(int j=0;j<n_mm;j++){
            M(j,i) = ls.calcMahDist(zs.segment<2>(2*j), state);
        }
    }

    Eigen::VectorXi assoc = solveCostMatrix(M);
    
    for(int i=0;i<n_mm;i++){
        // Check if it wants a new landmark
        if(assoc[i] >= n_lm){
            assoc[i] = -1;
            // Also check if there was any other landmarks semi-close to it
            // if so, don't make a new one
            for(int j=0;j<n_lm;j++){
                if(M(i,j) < beta){
                    assoc[i] = -2;
                }
            }
        }
    }
    return assoc;

}

void addLandmark(Eigen::Vector2d z, SE2_D& state){
    double x = state[0][0];
    double y = state[0][1];
    double phi = atan2(state()(1,0), state()(0,0));
    double r = z(0);
    double b = z(1);

    double xl = x + r*cos(b+phi);
    double yl = y + r*sin(b+phi);
    Eigen::Vector2d lm;
    lm << xl, yl;

    state.addCol(lm, Eigen::Matrix2d::Identity()*10000);
}

InEKF::SE2<> makeOdometry(Eigen::Vector2d u, double dt){
    double Ve = u(0);
    double alpha = u(1);

    double a = 3.78;
    double b = 0.50;
    double L = 2.83;
    double H = 0.76;

    double Vc = Ve / (1 - tan(alpha)*H/L);
    double motion_t = Vc/L*tan(alpha);
    double motion_x = Vc - Vc/L*tan(alpha)*b;
    double motion_y = Vc/L*tan(alpha)*a;

    return InEKF::SE2<>(dt*motion_t, dt*motion_x, dt*motion_y);
}

constexpr double pi() { return std::atan(1)*4; }

void loadEvents(std::vector<std::tuple<std::string,double,Eigen::VectorXd>> &events, std::string filename, std::string type){
    // Load file
    std::ifstream file(filename);

    // iterate through each line
    std::string temp;
    while(getline(file, temp)){
        // turn each line into a vector as file
        std::istringstream is( temp );
        std::vector<double> line( ( std::istream_iterator<double>( is ) ), ( std::istream_iterator<double>() ) );
        
        // turn each vector into an eigen vector
        auto n = line.size();
        Eigen::Map<Eigen::VectorXd> data(&line[1], n-1);

        // put into large array of data
        events.push_back( std::make_tuple(type, line[0], data) );
    }
}

int main() {
    /***** SETUP INITIAL STATE *****/
    Eigen::Matrix3d sig = Eigen::Matrix3d::Identity()*.1;
    Eigen::VectorXd init(3);
    init << 45*pi()/180, 0, 0;
    SE2_D x0(init, sig);

    /***** SETUP InEKF *****/
    InEKF::GPSSensor gps(3);
    InEKF::LandmarkSensor laser(0.5, 0.5*pi()/180);

    InEKF::InEKF<InEKF::OdometryProcessDynamic> iekf(x0, InEKF::RIGHT);
    iekf.addMeasureModel("GPS", &gps);
    iekf.addMeasureModel("Laser", &laser);
    Eigen::Vector3d Q;
    Q << 0.5*pi()/180, 0.05, 0.05;
    iekf.pModel->setQ(Q);

    /***** LOAD IN DATA *****/
    std::vector<std::tuple<std::string,double,Eigen::VectorXd>> events;
    loadEvents(events, "../data/victoria_park_ascii/DRS.txt", "odo");
    loadEvents(events, "../data/victoria_park_ascii/GPS.txt", "gps");
    loadEvents(events, "../data/victoria_park_ascii/LASER_TREE.txt", "laser");
    sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs){
        return std::get<1>(lhs) < std::get<1>(rhs);
    });
    double dt = 0;
    double last_t = 0;
    SE2_D s;

    /***** SETUP PLOT *****/
    auto f = plt::figure(false);
    auto ax = plt::gca();

    std::vector<double> empty = {0};
    std::vector<double> s_x = {0}, s_y = {0};
    std::vector<double> gps_x = {0}, gps_y = {0};

    auto lm_pts  = ax->scatter({-1,1}, {-1,1});
    plt::hold(true);
    auto traj = ax->plot(empty, empty);
    auto gps_pts = ax->scatter(empty, empty);
    plt::legend({lm_pts, traj, gps_pts}, {"Landmarks", "Vehicle", "GPS"});
    f->draw();

    /***** ITERATE THROUGH DATA *****/
    tqdm bar;
    int n=0, N=events.size();
    time_t last_plot = time(0);
    for(auto const& [e, t, data] : events){
        // Odometry
        if(e == "odo"){
            dt = t - last_t;
            last_t = t;

            InEKF::SE2 u = makeOdometry(data, dt);
            s = iekf.Predict(u, dt);
            s_x.push_back( s[0][0] );
            s_y.push_back( s[0][1] );
        }

        // GPS
        if(e == "gps"){
            s = iekf.Update(data, "GPS");
            gps_x.push_back(data[0]);
            gps_y.push_back(data[1]);
            s_x.back() = s[0][0];
            s_y.back() = s[0][1];
        }

        // Laser Measurement
        if(e == "laser"){
            // associate landmarks
            assert(data.size() % 2 == 0);
            int n_mm = data.size() / 2;
            Eigen::VectorXi assoc = dataAssocation(iekf.state_, data, laser);
            // iterate through them
            for(int i=0;i<n_mm;i++){
                if(assoc[i] == -1){
                    addLandmark(data.segment<2>(i*2), iekf.state_);
                    laser.sawLandmark(iekf.state_().cols()-2-1-1, iekf.state_);
                    iekf.Update(data.segment<2>(i*2),"Laser");
                }
                else if(assoc[i] != -2){
                    laser.sawLandmark(assoc[i], iekf.state_);
                    iekf.Update(data.segment<2>(i*2), "Laser");
                }

            }
        }

        // plot once a second
        if(difftime(time(0), last_plot) > 1){
            gps_pts->x_data(gps_x);
            gps_pts->y_data(gps_y);

            traj->x_data(s_x);
            traj->y_data(s_y);

            int n_lm = iekf.state_().cols()-2-1;
            Eigen::VectorXd lm_x = iekf.state_().block(0,3,1,n_lm-1).transpose();
            Eigen::VectorXd lm_y = iekf.state_().block(1,3,1,n_lm-1).transpose();
            lm_pts->x_data(std::vector<double>(lm_x.data(),n_lm+lm_x.data()));
            lm_pts->y_data(std::vector<double>(lm_y.data(),n_lm+lm_y.data()));
            last_plot = time(0);
        }

        bar.progress(n, N);
        n += 1;
        sleep(1e-8);
    }
    bar.finish();
    plt::show();

   return 0;
}
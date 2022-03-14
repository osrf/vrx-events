#include "Core/InEKF.h"
#include "iostream"

namespace InEKF {

template <class pM>
typename InEKF<pM>::Group InEKF<pM>::Predict(const U& u, double dt){    
    // Predict Sigma
    MatrixCov Sigma = state_.Cov();
    MatrixCov Phi = pModel->MakePhi(u, dt, state_, error_);

    MatrixCov Q = pModel->getQ();
    if(error_ == ERROR::RIGHT){
        MatrixCov Adj_X = Group::Ad( state_ );
        Q = Adj_X*Q*Adj_X.transpose();
    }
    Sigma += Q*dt;
    if(!Phi.isIdentity()){
        Sigma = Phi*Sigma*Phi.transpose();
    }
    state_.setCov( Sigma );

    // Predict mu
    state_ = pModel->f(u, dt, state_);

    return state_;
}


template <class pM>
typename InEKF<pM>::Group InEKF<pM>::Update(const Eigen::VectorXd& z, std::string type, MatrixH H){
    mModels[type]->setH(H);
    return Update(z, type);
}

template <class pM>
typename InEKF<pM>::Group InEKF<pM>::Update(const Eigen::VectorXd& z, std::string type){
    MeasureModel<Group> * m_model = mModels[type]; 

    // Do any preprocessing on z (fill it up, frame changes, etc)
    VectorB z_ = m_model->processZ(z, state_);;

    // Change H via adjoint if necessary
    MatrixH H = m_model->makeHError(state_, error_);

    // Use measurement model to make Sinv and V
    VectorV V = m_model->calcV(z_, state_);
    MatrixS Sinv = m_model->calcSInverse(state_);

    // Caculate K + dX
    MatrixK K = state_.Cov() * (H.transpose() * Sinv);    
    TangentVector K_V = K * V;

    // Apply to states
    if(error_ == ERROR::RIGHT){
        state_ = Group::Exp(K_V) * state_;
    }
    else{
        state_ = state_ * Group::Exp(K_V);
    }

    int size = state_.Cov().rows();
    MatrixCov I = MatrixCov::Identity(size, size);
    state_.setCov( state_.Cov() - K*(H*state_.Cov()) );

    return state_;
}

template <class pM>
void InEKF<pM>::addMeasureModel(std::string name, MeasureModel<Group>* m){
    mModels[name] = m;
}

template <class pM>
void InEKF<pM>::addMeasureModels(std::map<std::string, MeasureModel<Group>*> m){
    mModels.insert(m.begin(), m.end());
}

}
#pragma once

#include <Eigen/Core>

template<int Nsta, int Mobs>
class EKF_BASE{
public:

    EKF_BASE(){
        P.setZero();
        Q.setZero();
        R.setZero();

        G.setZero();
        F.setZero();
        H.setZero();
    }

    static const int n = Nsta;                /* number of state values */
    static const int m = Mobs;                /* number of observables */

    Eigen::Matrix<double, Nsta, 1> x;       /* state vector */
    
    Eigen::Matrix<double, Nsta, Nsta> P;    /* prediction error covariance */
    Eigen::Matrix<double, Nsta, Nsta> Q;    /* process noise covariance */
    Eigen::Matrix<double, Mobs, Mobs> R;    /* measurement error covariance */
    
    Eigen::Matrix<double, Nsta, Mobs> G;    /* Kalman gain; a.k.a. K */

    Eigen::Matrix<double, Nsta, Nsta> F;    /* Jacobian of process model */
    Eigen::Matrix<double, Mobs, Nsta> H;    /* Jacobian of measurement model */

    Eigen::Matrix<double, Nsta, Nsta> Pp;   /* P, post-prediction, pre-update */

    Eigen::Matrix<double, Nsta, 1> fx;                /* output of user defined f() state-transition function */
    Eigen::Matrix<double, Mobs, 1> hx;                /* output of user defined h() measurement function */
};

template<int Nsta, int Mobs>
class TinyEKF{

public:

    typedef EKF_BASE<Nsta, Mobs> ekf_t;
    
    typedef Eigen::Matrix<double, Nsta, 1> VectorN;
    typedef Eigen::Matrix<double, Mobs, 1> VectorM;

    typedef Eigen::Matrix<double, Nsta, Nsta> MatrixNN;
    typedef Eigen::Matrix<double, Mobs, Nsta> MatrixMN;

    TinyEKF(){ 
        x = ekf.x.data();
    }

    ~TinyEKF(){}  

    void setX(int i, double value){
        x[i] = value;
    }     
    
    double getX(int i) const {
        return x[i];
    }

    bool step(const double * z) 
    { 
        model(ekf.fx, ekf.F, ekf.hx, ekf.H); 

        const int n = ekf.n;
        const int m = ekf.m;

        Eigen::Map<const Eigen::Matrix<double, Nsta, 1>> measure(z);

        /* P_k = F_{k-1} P_{k-1} F^T_{k-1} + Q_{k-1} */
        ekf.Pp = ekf.F * ekf.P * (ekf.F.transpose()) + ekf.Q;

        /* G_k = P_k H^T_k (H_k P_k H^T_k + R)^{-1} */
        ekf.G = ekf.Pp * (ekf.H.transpose()) * ((ekf.H * ekf.P * (ekf.H.transpose()) + ekf.R).inverse());

        /* \hat{x}_k = \hat{x_k} + G_k(z_k - h(\hat{x}_k)) */
        ekf.x = ekf.fx + ekf.G * (measure - ekf.hx);

        /* P_k = (I - G_k H_k) P_k */
        ekf.P = (Eigen::Matrix<double, Nsta, Nsta>::Identity() - ekf.G * ekf.H) * ekf.Pp;

        return true;
    }

protected:


    virtual void model(VectorN& fx, MatrixNN& F, VectorM& hx, MatrixMN& H) = 0;

    inline void setP(int i, int j, double value){
        ekf.P(i, j) = value;
    }
    
    inline void setQ(int i, int j, double value){
        ekf.Q(i, j) = value;
    }

    inline void setR(int i, int j, double value){
        ekf.R(i, j) = value;
    }    

    
    
    double* x;

private:


    ekf_t ekf;


};


// These must be defined before including TinyEKF.h
#define EKF_N 2
#define EKF_M 2

#include <tinyekf.h>
#include <stdio.h>

static ekf_t _ekf;

#define EPS 1e-4

static const float Q[EKF_N*EKF_N] = {

    EPS, 0,
    0,   EPS
};

static const float R[EKF_M*EKF_M] = {

    EPS, 0,
    0,   EPS
};

// So process model Jacobian is identity matrix
static const float F[EKF_N*EKF_N] = {
    1, 0,
    0, 1
};

static const float H[EKF_M*EKF_N] = {
    1, 0,
    0, 1
};

int main(int argc, char ** argv)
{
    // Use identity matrix as initial covariance matrix
    float observed[EKF_N] = {1, 1};

    ekf_initialize(&_ekf, observed);

    for (int i = 0; i < 10; i++) {
        // add one to the observed state
        observed[0] += 1;
        observed[1] += 1;

        // Process model is f(x) = x
        const float fx[EKF_N] = { _ekf.x[0], _ekf.x[1] };

        // Run the prediction step of the DKF
        ekf_predict(&_ekf, fx, F, Q);

        const float hx[EKF_M] = { _ekf.x[0], _ekf.x[1] };

        // Run the update step
        ekf_update(&_ekf, observed, hx, H, R);

    }
    printf("Done\n");
    // compare the ekf and the true state
    printf("Final state vector: %f %f\n", _ekf.x[0], _ekf.x[1]);
    printf("True state vector: %f %f\n", observed[0], observed[1]);
    return 0;
}




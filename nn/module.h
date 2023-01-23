
// TODO: Use RealType?
class Module {
private:
public:
    virtual void foward(float *in, float *out) = 0;
    virtual void backward(float *in, float *out) = 0;
};
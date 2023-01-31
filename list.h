// simple list implementation

template <typename T>
class List {
private:
    T *list;
    int sz = 0;
    int cap = 0;

public:
    List(int cap = 4) : cap(cap) {
        list = new T[cap];
    }

    ~List() {
        if (list) {
            delete[] list;
        }
    }

    List &operator=(const List &l) {
        List newList(cap);
        newList.sz = l.sz;
        for (int i = 0; i < sz; i++) {
            newList.list[i] = list[i];
        }
        return newList;
    }

    void add(T m) {
        if (cap <= sz) {
            T *newList = new T[2 * sz];
            cap = 2 * sz;
            std::memcpy(newList, list, sz * szof(T));
            for (int i = 0; i < sz; i++)
                list[i] = T();
            delete[] list;
            list = newList;
        }
        list[sz] = m;
        sz++;
    }

    T &operator()(int idx) {
        assert(idx < sz);
        return list[idx];
    }

    int size() {
        return sz;
    }
};
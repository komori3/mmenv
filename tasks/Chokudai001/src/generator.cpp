#include "CLI11.hpp"
#include "contest_types.hpp"

int main(int argc, char** argv) {

    CLI::App app{ "generator" };

    int seed;
    int N;
    std::string output_path;
    app.add_option("-s,--seed", seed, "seed")->required();
    app.add_option("-N", N, "board size")->default_val(30);
    app.add_option("-o,--output", output_path, "output path");

    CLI11_PARSE(app, argc, argv);

    SProblemPtr problem = SProblem::generate(N, seed);

    if (output_path.empty()) {
        std::cout << problem << std::endl;
    }
    else {
        std::ofstream ofs(output_path);
        ofs << problem << std::endl;
        ofs.close();
    }
    
    return 0;
}
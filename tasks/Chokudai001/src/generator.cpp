#include "CLI11.hpp"
#include "contest_types.hpp"

int main(int argc, char** argv) {

    CLI::App app{ "generator" };

    int seed;
    int N;
    std::string output_path;
    app.add_option("-s,--seed", seed, "seed")->required();
    app.add_option("-N", N, "board size")->default_val(30);

    CLI11_PARSE(app, argc, argv);

    SProblemPtr problem = SProblem::generate(N, seed);

    std::cout << problem << std::endl;
    
    return 0;
}
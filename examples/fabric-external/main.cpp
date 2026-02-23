// Fabric SDK — External Architecture Example
//
// Usage: fabric-external-example <model.gguf> [prompt] [max_tokens]

#include <fabric/pipeline.h>
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt] [max_tokens]\n", argv[0]);
        return 1;
    }

    std::string prompt = (argc > 2) ? argv[2] : "The meaning of life is";
    int max_tokens     = (argc > 3) ? std::stoi(argv[3]) : 64;

    fabric::Pipeline pipe(argv[1]);

    pipe.generate(prompt, max_tokens, [](const char * piece) {
        printf("%s", piece);
        fflush(stdout);
    });

    printf("\n");
    return 0;
}

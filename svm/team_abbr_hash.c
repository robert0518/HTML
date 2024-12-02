#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int hash(char* str) {
    int sum = 0;
    for (int i = 0; i < 3; i++) {
        sum += str[i] * (i + 1);
    }
    sum %= 100;
    return sum;
}

int main() {
    FILE* file, * out;
    char line[100];

    file = fopen("team_abbr", "r");
    out = fopen("team_abbr_hash", "w");

    while (fgets(line, sizeof(line), file)) {
        *strrchr(line, '\n') = 0;
        fprintf(out, "%s %d\n", line, hash(line));
    }

    fclose(file);
    fclose(out);

    return 0;
}

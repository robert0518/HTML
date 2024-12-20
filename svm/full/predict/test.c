#include <stdio.h>
#include <string.h>

int main() {
    FILE* fp1 = fopen("full_output_gamma0.001_C100", "r");
    FILE* fp2 = fopen("full_output_gamma0.0001_C10", "r");
    FILE* out = fopen("compare", "w");
    char buffer[10];

    int i = 0;
    while (fgets(buffer, sizeof(buffer), fp1)) {
        *strchr(buffer, '\n') = 0;
        fprintf(out, "%d\t%s\t", i++, buffer);
        int temp = (buffer[0] == '1');

        fgets(buffer, sizeof(buffer), fp2);
        *strchr(buffer, '\n') = 0;
        fprintf(out, "%s\t%d\n", buffer, (temp == (buffer[0] == '1')));
    }

    fclose(fp1);
    fclose(fp2);
    fclose(out);

    return 0;
}
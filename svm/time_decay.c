#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpfr.h>
#define YEAR_NOW 2024
#define MONTH_NOW 1
#define DAY_NOW 1

double alpha[] = { 0.0001,0.0005,0.001 };

int date_dif(int year, int month, int day) {
    struct tm date1 = { 0 }, date2 = { 0 };
    time_t time1, time2;

    date1.tm_year = year - 1900;
    date1.tm_mon = month - 1;
    date1.tm_mday = day;

    date2.tm_year = YEAR_NOW - 1900;
    date2.tm_mon = MONTH_NOW - 1;
    date2.tm_mday = DAY_NOW;

    time1 = mktime(&date1);
    time2 = mktime(&date2);

    double difference = difftime(time2, time1) / (60 * 60 * 24);
    return (int)difference;
}

mpfr_t* decay(int day_dif, int a) {
    mpfr_t* ret = (mpfr_t*)malloc(sizeof(mpfr_t));
    mpfr_t tmp;
    mpfr_init2(*ret, 128);
    mpfr_init2(tmp, 128);
    mpfr_set_d(tmp, -1.0 * alpha[a] * day_dif, MPFR_RNDN);
    mpfr_exp(*ret, tmp, MPFR_RNDN);
    mpfr_clear(tmp);
    return ret;
}

int is_number(const char* str) {
    while (*str) {
        if (*str >= 65 && *str <= 122) { // between A and z
            return 0;
        }
        str++;
    }
    return 1;
}

int main() {
    for (int a = 0; a < 3; a++) {
        FILE* train = fopen("../train_data.csv", "r");
        char str[256];
        sprintf(str, "../train_data_decayed_alpha_%g.csv", alpha[a]);
        FILE* out = fopen(str, "w");

        char buffer[7000];

        fgets(buffer, sizeof(buffer), train);
        fprintf(out, "%s", buffer);

        // parse data
        char* name, * comma;
        while (fgets(buffer, sizeof(buffer), train)) {
            name = buffer;
            comma = strchr(name, ',');
            int num = 1, temp;
            mpfr_t* decay_coef;
            while (comma) {
                *comma = 0;

                if (num <= 3) {
                    fprintf(out, "%s,", name);
                } else if (num == 4) {
                    fprintf(out, "%s,", name);
                    char* year = name;
                    char* month = strchr(year, '-');
                    *month = 0;
                    month++;
                    char* day = strchr(month, '-');
                    *day = 0;
                    day++;
                    int day_dif = date_dif(atoi(year), atoi(month), atoi(day));
                    decay_coef = decay(day_dif, a);
                } else if (num <= 13) {
                    fprintf(out, "%s,", name);
                } else {
                    if (name != comma) {
                        if (is_number(name)) {
                            mpfr_t tmp;
                            mpfr_init2(tmp, 128);
                            mpfr_set_str(tmp, name, 10, MPFR_RNDN);
                            mpfr_mul(tmp, tmp, *decay_coef, MPFR_RNDN);
                            mpfr_fprintf(out, "%.20Rf,", tmp);
                            mpfr_clear(tmp);
                        } else {
                            fprintf(out, "%s,", name);
                        }
                    } else { fprintf(out, ","); }
                }
                name = comma + 1;
                comma = strchr(name, ',');
                num++;
            }
            comma = strchr(name, '\n');
            *comma = 0;
            if (name != comma && is_number(name)) {
                mpfr_t tmp;
                mpfr_init2(tmp, 128);
                mpfr_set_str(tmp, name, 10, MPFR_RNDN);
                mpfr_mul(tmp, tmp, *decay_coef, MPFR_RNDN);
                mpfr_fprintf(out, "%.20Rf", tmp);
                mpfr_clear(tmp);
            }
            fprintf(out, "\n");
            mpfr_clear(*decay_coef);
        }

        fclose(train);
        fclose(out);
    }


    return 0;
}
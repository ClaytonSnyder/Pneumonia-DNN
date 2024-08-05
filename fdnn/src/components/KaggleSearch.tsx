import { Divider, Typography } from "@mui/material";
import TextField from "@mui/material/TextField";
import * as React from "react";
import { KaggleDatasetsTable } from "./KaggleDatasetsTable";

interface KaggleSearchProps {
    setSelectedDatasets: (identifier: string, url: string) => void;
}

interface DebouncedFunction<T extends (...args: any[]) => void> extends Function {
    (...args: Parameters<T>): void;
    cancel: () => void;
}

function debounce<T extends (...args: any[]) => void>(func: T, wait: number): DebouncedFunction<T> {
    let timeout: ReturnType<typeof setTimeout> | null = null;

    const debouncedFunction = ((...args: Parameters<T>) => {
        const later = () => {
            timeout = null;
            func(...args);
        };

        if (timeout !== null) {
            clearTimeout(timeout);
        }
        timeout = setTimeout(later, wait);
    }) as DebouncedFunction<T>;

    debouncedFunction.cancel = () => {
        if (timeout !== null) {
            clearTimeout(timeout);
            timeout = null;
        }
    };

    return debouncedFunction;
}

export interface KaggleDataset {
    description: string;
    creatorName: string;
    lastUpdated: string;
    licenseName: string;
    ownerName: string;
    identifier: string;
    subtitle: string;
    url: string;
    usabilityRating: number;
    voteCount: number;
    totalBytes: number;
}

export function KaggleSearch(props: KaggleSearchProps) {
    const [inputValue, setInputValue] = React.useState<string>("");
    const [kaggleDatasets, setKaggleDatasets] = React.useState<Array<KaggleDataset>>([]);

    // Function to be called when user stops typing
    const fetchData = async (query: string) => {
        if (!query) return;
        fetch(`/api/dataset/kaggle?query=${query}`).then((res) =>
            res.json().then((data: Array<KaggleDataset>) => {
                setKaggleDatasets(data);
            })
        );
    };

    // Use debounce function
    const debouncedFetchData = debounce(fetchData, 500);

    React.useEffect(() => {
        debouncedFetchData(inputValue);
        return () => {
            if (debouncedFetchData.cancel) debouncedFetchData.cancel();
        };
    }, [inputValue]);

    return (
        <div style={{ marginTop: 15 }}>
            <TextField
                fullWidth
                label="Search for kaggle datasets"
                variant="outlined"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
            />
            <div style={{ marginTop: 20, height: 370 }}>
                <Typography gutterBottom variant="h6">
                    Kaggle Datasets
                </Typography>
                <Divider />
                {kaggleDatasets.length == 0 ? (
                    <Typography gutterBottom variant="body1">
                        None
                    </Typography>
                ) : (
                    <KaggleDatasetsTable datasets={kaggleDatasets} setSelectedDatasets={props.setSelectedDatasets} />
                )}
            </div>
        </div>
    );
}

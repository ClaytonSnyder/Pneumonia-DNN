import { Link } from "react-router-dom";
import { SelectedKaggleDataset } from "./KaggleAddWizard";
import {
    Alert,
    Box,
    Button,
    FormControl,
    InputLabel,
    MenuItem,
    Select,
    SelectChangeEvent,
    Typography,
} from "@mui/material";

interface KaggleMetadataSelectionProps {
    dataset: SelectedKaggleDataset;
    next: () => void;
    back: () => void;
    cancel: () => void;
    setCsv: (csvFile: string) => void;
    csv: string;
    setIsLoading: (isLoading: boolean) => void;
    setColumns: (columns: Array<string>) => void;
}

interface ColumnsResponse {
    columns: Array<string>;
}
export function KaggleMetadataSelection(props: KaggleMetadataSelectionProps) {
    const { dataset, next, back, cancel, setCsv, csv, setIsLoading, setColumns } = props;

    const onNext = () => {
        setIsLoading(true);
        fetch(`/api/dataset/columns?dataset_identifier=${dataset.identifier}&path_to_metadata=${csv}`)
            .then(function (res) {
                return res.json();
            })
            .then(function (response: ColumnsResponse) {
                setColumns(response.columns);
                next();
                setIsLoading(false);
            });
    };

    const handleChange = (event: SelectChangeEvent) => {
        setCsv(event.target.value as string);
    };

    return (
        <div>
            <Alert severity="success">
                <Typography gutterBottom variant="h6">
                    {dataset.identifier} - <Link to={dataset.url}>{dataset.url}</Link>
                </Typography>
            </Alert>
            <div style={{ marginTop: 15 }}>
                <Typography variant="body1">
                    Please, select the CSV in this dataset that contains the label label metadata for the images.
                </Typography>
            </div>
            <FormControl fullWidth style={{ marginTop: 15 }}>
                <InputLabel id="demo-simple-select-label">CSV</InputLabel>
                <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={csv}
                    label="CSV"
                    onChange={handleChange}
                >
                    <MenuItem value={""}>None</MenuItem>
                    {props.dataset.metadata.csv.map((csv) => (
                        <MenuItem key={csv} value={csv}>
                            {csv}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
            <Box sx={{ display: "flex", flexDirection: "row", pt: 2 }}>
                <Button color="inherit" onClick={back} sx={{ mr: 1 }}>
                    Back
                </Button>
                <Button disabled={csv === ""} onClick={onNext}>
                    Next
                </Button>
                <Box sx={{ flex: "1 1 auto" }} />
                <Button onClick={cancel}>Cancel</Button>
            </Box>
        </div>
    );
}

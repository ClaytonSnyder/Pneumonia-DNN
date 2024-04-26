import { Button, TableCell, TableRow } from "@mui/material";
import { KaggleDataset } from "./KaggleSearch";

interface KaggleDatasetsTableRowProps {
    row: KaggleDataset;
    setSelectedDatasets: (identifier: string, url: string) => void;
}
export function KaggleDatasetsTableRow(props: KaggleDatasetsTableRowProps) {
    const { row } = props;

    const onSelect = () => {
        props.setSelectedDatasets(row.identifier, row.url);
    };

    return (
        <TableRow key={row.identifier} sx={{ "&:last-child td, &:last-child th": { border: 0 } }}>
            <TableCell component="th" scope="row">
                <Button onClick={onSelect}>Select</Button>
            </TableCell>
            <TableCell component="th" scope="row">
                {row.identifier}
            </TableCell>
            <TableCell align="right">{row.subtitle}</TableCell>
            <TableCell align="right">{row.creatorName}</TableCell>
            <TableCell align="right">{row.lastUpdated.replace(" GMT", "")}</TableCell>
            <TableCell align="right">{Math.ceil(row.usabilityRating * 100)}%</TableCell>
            <TableCell align="right">{row.voteCount}</TableCell>
        </TableRow>
    );
}
